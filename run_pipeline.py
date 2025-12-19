#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys


def run_cmd(cmd, **kwargs):
	"""Run a command, print it first, and fail loudly on error."""
	print("+", " ".join(map(str, cmd)))
	subprocess.run(cmd, check=True, **kwargs)


def run_reviewer_on_bams(args):
	bam_dir = Path(args.bam_dir)
	vcf_dir = Path(args.vcf_dir)
	reference = Path(args.reference)
	catalog = Path(args.catalog)
	out_dir = Path(args.output_dir)

	if not bam_dir.is_dir():
		print(f"ERROR: BAM directory not found: {bam_dir}", file=sys.stderr)
		sys.exit(1)
	if not vcf_dir.is_dir():
		print(f"ERROR: VCF directory not found: {vcf_dir}", file=sys.stderr)
		sys.exit(1)
	if not reference.is_file():
		print(f"ERROR: Reference FASTA not found: {reference}", file=sys.stderr)
		sys.exit(1)
	if not catalog.is_file():
		print(f"ERROR: Catalog JSON not found: {catalog}", file=sys.stderr)
		sys.exit(1)

	out_dir.mkdir(parents=True, exist_ok=True)

	bam_files = sorted(bam_dir.glob("*.bam"))
	if not bam_files:
		print(f"No .bam files found in {bam_dir}", file=sys.stderr)
		sys.exit(1)

	print(f"Found {len(bam_files)} BAM files in {bam_dir}")
	print(f"Writing REViewer outputs to: {out_dir}")

	for bam_path in bam_files:
		basename = bam_path.stem  # e.g. "BEAN1_HG00684"
		if "_" not in basename:
			print(f"Skipping {bam_path} (cannot parse locus/sample from name)")
			continue

		parts = basename.split("_")

		if len(parts) < 4:
			print(f"Skipping {bam_path} (bad naming format)")
			continue

		locus = "_".join(parts[0:3])     # chr1_10000_10126
		sample = parts[3]               # HG002

		bai_path = bam_path.with_suffix(".bam.bai")
		if not bai_path.exists():
			# Make sure BAM is indexed
			run_cmd([args.samtools_bin, "index", str(bam_path)])

		vcf_path = vcf_dir / f"HG002.vcf"
		if not vcf_path.is_file():
			print(f"WARNING: skipping {bam_path} (VCF not found: {vcf_path})")
			continue

		out_prefix = out_dir / basename
		cmd = [
			args.reviewer_bin,
			"--reads", str(bam_path),
			"--vcf", str(vcf_path),
			"--reference", str(reference),
			"--catalog", str(catalog),
			"--locus", locus,
			"--output-prefix", str(out_prefix),
		]
		run_cmd(cmd)

	print("\n REViewer run complete.")


def run_flipbook(args):
	out_dir = Path(args.output_dir)

	if args.flipbook_mode == "none":
		print("Flipbook step skipped (flipbook-mode=none).")
		return

	if not out_dir.is_dir():
		print(f"ERROR: Output directory for flipbook not found: {out_dir}", file=sys.stderr)
		sys.exit(1)

	print(f"\nStarting Flipbook in mode '{args.flipbook_mode}' on {out_dir}")

	if args.flipbook_mode == "server":
		cmd = [
			sys.executable, "-m", "flipbook",
			str(out_dir),
			"--port", str(args.flipbook_port),
		]
		run_cmd(cmd)
	elif args.flipbook_mode == "static":
		cmd = [
			sys.executable, "-m", "flipbook",
			"--generate-static-website",
			str(out_dir),
		]
		run_cmd(cmd)
		print("\n Static website generated in that directory.")

def main():
	parser = argparse.ArgumentParser(
		description="Run REViewer on BAM/VCF pairs and then browse outputs with Flipbook."
	)
	parser.add_argument("--reviewer-bin", default="REViewer",
						help="Path to REViewer executable (default: REViewer in $PATH)")
	parser.add_argument("--samtools-bin", default="samtools",
						help="Path to samtools executable (default: samtools in $PATH)")

	parser.add_argument("--bam-dir", required=True,
						help="Directory with .bam files (e.g. reviewer/tests/inputs/bamlets)")
	parser.add_argument("--vcf-dir", required=True,
						help="Directory with .vcf files (e.g. reviewer/tests/inputs/vcfs)")
	parser.add_argument("--reference", required=True,
						help="Reference FASTA (e.g. reviewer/tests/inputs/genomes/HG38_chr16.fa)")
	parser.add_argument("--catalog", required=True,
						help="STR catalog JSON (e.g. reviewer/tests/inputs/catalogs/stranger_variant_catalog_hg38_chr16.json)")
	parser.add_argument("--output-dir", default="reviewer/tests/flipbook_images",
						help="Directory to write REViewer outputs (SVG/TSV). Default: reviewer/tests/flipbook_images")

	parser.add_argument("--flipbook-mode",
						choices=["server", "static", "none"],
						default="server",
						help="How to run Flipbook: 'server' (default), 'static', or 'none'.")
	parser.add_argument("--flipbook-port", type=int, default=8080,
						help="Port for Flipbook when flipbook-mode=server (default: 8080).")

	args = parser.parse_args()

	run_reviewer_on_bams(args)
	run_flipbook(args)


if __name__ == "__main__":
	main()
