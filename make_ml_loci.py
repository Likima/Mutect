#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="predictions_strs_only.json")
    parser.add_argument("--output", required=True, help="ml_loci_for_reviewer.json")
    args = parser.parse_args()

    with open(args.predictions, "r") as f:
        preds = json.load(f)

    loci = []
    for i, entry in enumerate(preds):
        seq = entry.get("sequence", "")
        if not seq:
            continue
        chrom = (entry.get("chromosome") or entry.get("reference_name")
                 or entry.get("chr") or "unknown")
        start = (entry.get("position") or entry.get("reference_start")
                 or entry.get("start") or 0)
        start = int(start)
        probability = entry.get("str_probability", 0.0)

        loci.append({
            "locusId": f"locus_{i+1}",
            "chrom": chrom,
            "start": start,
            "end": start + len(seq),
            "repeatSequence": seq,
            "probability": probability,
            "readName": entry.get("query_name", entry.get("read_name", "unknown"))
        })

    out = {"loci": loci}

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved REViewer ML-loci file to: {args.output}")
    print(f"Total loci: {len(loci)}")

if __name__ == "__main__":
    main()
