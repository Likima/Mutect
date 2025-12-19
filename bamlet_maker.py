#!/usr/bin/env python3
import json
import argparse
import pysam
from pathlib import Path
import os

DEFAULT_WINDOW = 300   # +/- bp window around locus start/end


def load_loci(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["loci"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_bamlet(bam_path, out_bam_path, chrom, start, end):
    """
    Create a BAMlet: a small BAM file containing reads overlapping [start, end].
    Handles chr vs non-chr contig naming.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")

    # Resolve contig name (e.g., 1 -> chr1 if needed)
    refs = list(bam.references)
    chrom_resolved = chrom

    if chrom_resolved not in refs:
        # Try adding 'chr'
        if not chrom_resolved.startswith("chr") and ("chr" + chrom_resolved) in refs:
            chrom_resolved = "chr" + chrom_resolved
        # Try stripping 'chr'
        elif chrom_resolved.startswith("chr") and chrom_resolved[3:] in refs:
            chrom_resolved = chrom_resolved[3:]
        else:
            bam.close()
            raise ValueError(f"Chromosome '{chrom}' not found in BAM references: {refs[:10]}...")

    # Create output BAMlet with same header
    out_bam = pysam.AlignmentFile(out_bam_path, "wb", template=bam)

    # Write reads
    for read in bam.fetch(chrom_resolved, start, end):
        out_bam.write(read)

    out_bam.close()
    bam.close()

    # Sort and index the BAMlet
    sorted_path = out_bam_path.replace(".bam", ".sorted.bam")
    pysam.sort("-o", sorted_path, out_bam_path)
    pysam.index(sorted_path)

    # Replace the unsorted with sorted
    os.remove(out_bam_path)
    os.rename(sorted_path, out_bam_path)



def main():
    parser = argparse.ArgumentParser(description="Generate BAMlets for REViewer")
    parser.add_argument("--bam", required=True, help="Input BAM file")
    parser.add_argument("--loci", required=True, help="ml_loci_for_reviewer.json")
    parser.add_argument("--outdir", default="bamlets", help="Output directory for BAMlets")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Window size around locus (default 300bp)")
    args = parser.parse_args()

    loci = load_loci(args.loci)
    ensure_dir(args.outdir)

    print(f"Loaded {len(loci)} loci")
    print(f"Writing BAMlets to: {args.outdir}")
    print("-----------------------------------------------------")

    for locus in loci:
        locus_id = locus["locusId"]
        chrom = str(locus["chrom"])
        start = int(locus["start"])
        end = int(locus["end"])

        w_start = max(0, start - args.window)
        w_end = end + args.window

        out_bam_path = os.path.join(args.outdir, f"{locus_id}.bam")

        print(f"[{locus_id}] {chrom}:{w_start}-{w_end} → {out_bam_path}")

        create_bamlet(
            bam_path=args.bam,
            out_bam_path=out_bam_path,
            chrom=chrom,
            start=w_start,
            end=w_end
        )

    print("\nAll BAMlets generated successfully!")


if __name__ == "__main__":
    main()
