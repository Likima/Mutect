from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory, abort

try:
	# Optional; enable when available
	from flask_cors import CORS  # type: ignore
except Exception:  # pragma: no cover
	CORS = None  # type: ignore


def get_project_root() -> Path:
	"""Resolve repository root from this file path."""
	# web/src/backend.py → project root is two levels up
	return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"

# Allow overriding directories via env vars
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", str(DEFAULT_FIGURES_DIR)))

app = Flask(__name__, static_folder=None)
if CORS:
	CORS(app)


def _load_json(path: Path) -> Any:
	if not path.exists():
		abort(404, description=f"File not found: {path}")
	with path.open("r") as f:
		return json.load(f)


def _coerce_bool(value: Optional[str], default: bool = False) -> bool:
	if value is None:
		return default
	return value.lower() in {"1", "true", "t", "yes", "y"}


def _prediction_paths(only_strs: bool) -> Path:
	base = OUTPUT_DIR / "predictions.json"
	if only_strs:
		return Path(str(base).replace(".json", "_strs_only.json"))
	return base


def _normalize_location(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
	"""Best-effort harmonization of chromosome and position keys."""
	chrom = item.get("chromosome") or item.get("reference_name") or item.get("chrom") or item.get("chr")
	pos = item.get("position") or item.get("reference_start") or item.get("start")
	try:
		pos = int(pos) if pos is not None else None
	except (TypeError, ValueError):
		pos = None
	return chrom, pos


@app.get("/health")
def health() -> Any:
	return {"status": "ok", "output_dir": str(OUTPUT_DIR), "figures_dir": str(FIGURES_DIR)}


@app.get("/api/summary")
def api_summary() -> Any:
	summary_path = OUTPUT_DIR / "predictions_summary.json"
	return jsonify(_load_json(summary_path))


@app.get("/api/predictions")
def api_predictions() -> Any:
	"""
	Query params:
	- only_strs: bool (default true)
	- min_prob: float (0..1)
	- motif: str (substring match)
	- chromosome: str
	- has_repeat: bool
	- page: int (1-based)
	- page_size: int
	"""
	only_strs = _coerce_bool(request.args.get("only_strs"), default=True)
	min_prob = request.args.get("min_prob", type=float)
	motif = request.args.get("motif")
	chromosome_filter = request.args.get("chromosome")
	has_repeat = request.args.get("has_repeat")
	has_repeat_bool: Optional[bool] = None
	if has_repeat is not None:
		has_repeat_bool = _coerce_bool(has_repeat, default=None)  # type: ignore

	sort_by = (request.args.get("sort_by") or "").lower()
	sort_dir = (request.args.get("sort_dir") or "desc").lower()
	if sort_by and sort_by not in {"pos", "prob", "motif", "count", "len", "chrom"}:
		abort(400, description="Invalid sort_by")
	if sort_dir not in {"asc", "desc"}:
		abort(400, description="Invalid sort_dir")

	page = max(request.args.get("page", default=1, type=int) or 1, 1)
	page_size = min(max(request.args.get("page_size", default=100, type=int) or 100, 1), 1000)

	data_path = _prediction_paths(only_strs=only_strs)
	records: List[Dict[str, Any]] = _load_json(data_path)

	def passes_filters(rec: Dict[str, Any]) -> bool:
		if min_prob is not None and float(rec.get("str_probability", 0.0)) < min_prob:
			return False
		if motif:
			rm = str(rec.get("repeat_motif", "") or "")
			if motif.upper() not in rm.upper():
				return False
		if chromosome_filter:
			chrom, _ = _normalize_location(rec)
			if not chrom or chrom != chromosome_filter:
				return False
		if has_repeat_bool is not None:
			if bool(rec.get("has_repeat", False)) != has_repeat_bool:
				return False
		return True

	filtered = [r for r in records if passes_filters(r)]

	def sort_key(rec: Dict[str, Any]) -> Any:
		chrom, pos = _normalize_location(rec)
		if sort_by == "pos":
			return (pos is None, pos if pos is not None else -1)
		if sort_by == "prob":
			return float(rec.get("str_probability", 0.0))
		if sort_by == "motif":
			return str(rec.get("repeat_motif", "") or "").upper()
		if sort_by == "count":
			try:
				return int(rec.get("repeat_count", 0) or 0)
			except Exception:
				return 0
		if sort_by == "len":
			try:
				return int(rec.get("repeat_length", 0) or 0)
			except Exception:
				return 0
		if sort_by == "chrom":
			return str(chrom or "")
		# default: probability
		return float(rec.get("str_probability", 0.0))

	# Apply sorting (default prob desc)
	if not sort_by:
		filtered.sort(key=lambda r: float(r.get("str_probability", 0.0)), reverse=True)
	else:
		filtered.sort(key=sort_key, reverse=(sort_dir == "desc"))

	total = len(filtered)
	start_idx = (page - 1) * page_size
	end_idx = start_idx + page_size
	page_items = filtered[start_idx:end_idx]

	# augment normalized fields for UI convenience
	for rec in page_items:
		chrom, pos = _normalize_location(rec)
		rec.setdefault("_chromosome", chrom)
		rec.setdefault("_position", pos)

	return jsonify(
		{
			"total": total,
			"page": page,
			"page_size": page_size,
			"items": page_items,
		}
	)


@app.get("/api/loci")
def api_loci() -> Any:
	"""
	Generate REViewer-like loci JSON on the fly from STR-only predictions.
	Output shape: { "loci": [ { locusId, chrom, start, end, repeatSequence, probability, readName } ] }
	"""
	preds_path = _prediction_paths(only_strs=True)
	preds: List[Dict[str, Any]] = _load_json(preds_path)

	loci: List[Dict[str, Any]] = []
	for i, entry in enumerate(preds):
		sequence = entry.get("sequence", "")
		chrom, start = _normalize_location(entry)
		if chrom is None or start is None or not sequence:
			# Skip entries without location info
			continue
		loci.append(
			{
				"locusId": f"locus_{i+1}",
				"chrom": chrom,
				"start": start,
				"end": start + len(sequence),
				"repeatSequence": sequence,
				"probability": float(entry.get("str_probability", 0.0)),
				"readName": entry.get("query_name", "unknown"),
				"repeatMotif": entry.get("repeat_motif", "N/A"),
				"repeatCount": entry.get("repeat_count", 0),
				"repeatLength": entry.get("repeat_length", 0),
			}
		)

	return jsonify({"count": len(loci), "loci": loci})


@app.get("/api/figures")
def api_figures() -> Any:
	"""
	List available figure PNGs for the dashboard grouped by subfolder.
	Returns relative URLs under /figures/* for direct image access.
	"""
	if not FIGURES_DIR.exists():
		return jsonify({"groups": {}, "base": "/figures"})

	groups: Dict[str, List[str]] = {}
	for sub in sorted([p for p in FIGURES_DIR.iterdir() if p.is_dir()]):
		images = [f"/figures/{sub.name}/{img.name}" for img in sorted(sub.glob("*.png"))]
		if images:
			groups[sub.name] = images
	# top-level images
	top_images = [f"/figures/{img.name}" for img in sorted(FIGURES_DIR.glob("*.png"))]
	if top_images:
		groups["_root"] = top_images

	return jsonify({"groups": groups, "base": "/figures"})


@app.get("/figures/<path:filename>")
def serve_figure(filename: str) -> Any:
	# Try subdirectories transparently; send_from_directory handles security checks
	full = FIGURES_DIR / filename
	if not full.exists():
		abort(404)
	# Determine directory and actual file name
	return send_from_directory(full.parent, full.name)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5001"))
	app.run(host="0.0.0.0", port=port, debug=True)


