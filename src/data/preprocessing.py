def summarize_variants(variants):
    counts = {}
    for v in variants:
        significance = v.get("clinical_significance", "N/A") ##Gets the Classification/Review status
        if isinstance(significance, list):
            for item in significance:
                counts[item] = counts.get(item, 0) + 1
        else:
            counts[significance] = counts.get(significance, 0) + 1

    for k, v in counts.items():
        print(f"  - {k}: {v}")

def _coerce_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return None


def _pick_best_placement(placements):
    """
    placements can be a list or a dict. Prefer GRCh38, then GRCh37, else first.
    Each placement is expected to have chr_start/chr_end or start/stop-ish fields.
    """
    if not placements:
        return None
    if isinstance(placements, dict):
        placements = [placements]

    def rank(p):
        asm = (p.get("assembly") or "").upper()
        if asm == "GRCH38":
            return 0
        if asm == "GRCH37":
            return 1
        return 2

    return sorted(placements, key=rank)[0]


def _extract_start_end_from_obj(obj):
    """
    Given a dict that might represent a placement or a location record,
    pull out the best available start/end pair and return (s, e) as ints or (None, None).
    """
    cand_start_keys = ("chr_start", "start", "position")
    cand_end_keys   = ("chr_end", "end", "stop", "chr_stop")

    s = e = None
    for k in cand_start_keys:
        if k in obj:
            s = _coerce_int(obj.get(k))
            if s is not None:
                break
    for k in cand_end_keys:
        if k in obj:
            e = _coerce_int(obj.get(k))
            if e is not None:
                break
    return (s, e)


def _get_variant_length(variant):
    """Estimate variant length in base pairs from common fields.

    Heuristics tried (in order):
    1) dbVar docsum placements: variant["dbvarplacementlist"] (prefer GRCh38)
    2) variation_set -> variation_loc[0] -> start/stop (older/alternative layout)
    3) Top-level numeric fields: svlen, length, variant_length, size, ins_length
    Returns integer length if determinable, otherwise None.
    """

    # --- 1) dbVar docsum coordinates (dbvarplacementlist) ---
    try:
        placements = variant.get("dbvarplacementlist")
        best = _pick_best_placement(placements)
        if best:
            s, e = _extract_start_end_from_obj(best)
            if s is not None and e is not None:
                return abs(e - s) + 1
    except Exception:
        pass

    try:
        s, e = _extract_start_end_from_obj(variant)
        if s is not None and e is not None:
            return abs(e - s) + 1
    except Exception:
        pass

    return None


def filter_str_by_min_length(variants, min_bp=50, include_unknown=False):
    """Filter a list of dbVar variant dicts to STRs with length >= min_bp."""
    kept = []
    filtered_out = 0

    for v in variants:
        length = _get_variant_length(v)
        if length is None:
            if include_unknown:
                kept.append(v)
            else:
                filtered_out += 1
        elif length >= min_bp:
            kept.append(v)
        else:
            filtered_out += 1

    print(
        f"filter_str_by_min_length: kept={len(kept)}, filtered_out={filtered_out}, "
        f"min_bp={min_bp}, include_unknown={include_unknown}"
    )
    return kept
