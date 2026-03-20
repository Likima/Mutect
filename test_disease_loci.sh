#!/usr/bin/env bash
# test_disease_loci.sh — Test the STR pipeline against known pathogenic TR loci
#
# Tests samples from the 1000 Genomes ONT dataset and GIAB HG002
# against known disease-associated tandem repeat expansion loci.
#
# Usage:
#   ./test_disease_loci.sh              # Run all tests
#   ./test_disease_loci.sh atxn10       # Run only the ATXN10 test
#   ./test_disease_loci.sh hg002        # Run only the HG002 tests
#   ./test_disease_loci.sh htt          # Only HTT (Huntington)
#   ./test_disease_loci.sh fmr1         # Only FMR1 (Fragile X)
#   ./test_disease_loci.sh fxn          # Only FXN (Friedreich ataxia)
#   ./test_disease_loci.sh dmpk         # Only DMPK (Myotonic dystrophy)
#   ./test_disease_loci.sh rfc1         # Only RFC1 (CANVAS)
#
# Prerequisites:
#   - uv (Python package manager)
#   - Training data: output/str_variants.json, output/normal_sequences.json
#
# Simplified one-liner (what this script does per test):
#   uv run main.py --model-path output/str_model.joblib \
#     --predict-bam <BAM> --predict-chr <CHR> \
#     --predict-start <START> --predict-end <END> \
#     --output-dir output/disease_tests/<NAME>

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

STR_DATA="output/str_variants.json"
NORMAL_DATA="output/normal_sequences.json"
OUTPUT_BASE="output/disease_tests"
MODEL_PATH="output/str_model.joblib"

# ── Helper ──────────────────────────────────────────────────────────────────

run_test() {
    local name="$1"
    local bam="$2"
    local chr="$3"
    local start="$4"
    local end="$5"
    local disease="$6"
    local outdir="${OUTPUT_BASE}/${name}"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  TEST: ${name}${NC}"
    echo -e "${CYAN}  Disease: ${disease}${NC}"
    echo -e "${CYAN}  Region: ${chr}:${start}-${end}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    mkdir -p "$outdir"

    # Use pre-trained model if available, otherwise train
    if [ -f "$MODEL_PATH" ]; then
        echo -e "${GREEN}Using pre-trained model: ${MODEL_PATH}${NC}"
        uv run main.py \
            --model-path "$MODEL_PATH" \
            --predict-bam "$bam" \
            --predict-chr "$chr" \
            --predict-start "$start" \
            --predict-end "$end" \
            --output-dir "$outdir" \
            2>&1 | tail -30
    else
        echo -e "${YELLOW}No pre-trained model found. Training + predicting...${NC}"
        uv run main.py \
            --str-data "$STR_DATA" \
            --normal-data "$NORMAL_DATA" \
            --train \
            --predict-bam "$bam" \
            --predict-chr "$chr" \
            --predict-start "$start" \
            --predict-end "$end" \
            --output-dir "$outdir" \
            2>&1 | tail -30
    fi

    # Show results summary
    if [ -f "${outdir}/predictions_summary.json" ]; then
        echo ""
        echo -e "${GREEN}Results for ${name}:${NC}"
        python3 -c "
import json, sys
with open('${outdir}/predictions_summary.json') as f:
    s = json.load(f)
print(f\"  Total sequences: {s['total_sequences']}\")
print(f\"  Predicted STRs:  {s['predicted_strs']}\")
print(f\"  Threshold:       {s['threshold']}\")
ca = s.get('clinical_annotations', {})
if ca.get('total_disease_matches', 0) > 0:
    print(f\"  Disease matches: {ca['total_disease_matches']}\")
    for d, c in ca.get('disease_matches', {}).items():
        print(f\"    - {d}: {c}\")
top = s.get('top_20_str_predictions', [])
for i, p in enumerate(top[:5]):
    motif = p.get('repeat_motif', 'N/A')
    prob = p.get('probability', 0)
    disease = p.get('redatlas_disease') or ''
    ds = f' [{disease}]' if disease else ''
    print(f\"  #{i+1}: motif={motif}, prob={prob:.4f}{ds}\")
" 2>/dev/null || echo -e "${YELLOW}  (Could not parse summary)${NC}"
    fi

    echo -e "${GREEN}  Output: ${outdir}/${NC}"
}

# ── Ensure training data exists ─────────────────────────────────────────────

if [ ! -f "$STR_DATA" ] || [ ! -f "$NORMAL_DATA" ]; then
    echo -e "${RED}Training data not found:${NC}"
    echo "  $STR_DATA"
    echo "  $NORMAL_DATA"
    echo "Run the default pipeline first to generate training data."
    exit 1
fi

# ── Train model once if not already trained ─────────────────────────────────

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Training model (one-time)...${NC}"
    uv run main.py \
        --str-data "$STR_DATA" \
        --normal-data "$NORMAL_DATA" \
        --train \
        --output-dir output
    echo -e "${GREEN}Model trained and saved to ${MODEL_PATH}${NC}"
fi

# ── Define test cases ───────────────────────────────────────────────────────

# 1000 Genomes ONT S3 base URL
ONT_BASE="https://1000g-ont.s3.amazonaws.com/ALIGNMENT_AND_ASSEMBLY_DATA/FIRST_100/NAPU_PIPELINE/HG38"

# GIAB HG002 PacBio base URL
HG002_BAM="https://downloads.pacbcloud.com/public/dataset/HG002-CpG-methylation-202202/HG002.GRCh38.haplotagged.bam"

# Local BAM if available
HG01122_LOCAL="HG01122_ATXN10.bam"

TEST_FILTER="${1:-all}"

mkdir -p "$OUTPUT_BASE"

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  STR Disease Loci Test Suite${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

# ── Test 1: ATXN10 — HG01122 (SCA10, >1000 ATTCT repeats) ──────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "atxn10" ]]; then
    if [ -f "$HG01122_LOCAL" ]; then
        run_test "HG01122_ATXN10_local" \
            "$HG01122_LOCAL" \
            "chr22" 45790000 45800000 \
            "Spinocerebellar ataxia 10 (ATTCT expansion)"
    else
        run_test "HG01122_ATXN10_remote" \
            "${ONT_BASE}/HG01122-ONT-hg38-R9-LSK110-guppy-sup-5mC/HG01122-ONT-hg38-R9-LSK110-guppy-sup-5mC.PMDV_FINAL.haplotagged.bam" \
            "chr22" 45790000 45800000 \
            "Spinocerebellar ataxia 10 (ATTCT expansion)"
    fi
fi

# ── Test 2: HG002 — HTT locus (Huntington, normal allele) ──────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" || "$TEST_FILTER" == "htt" ]]; then
    run_test "HG002_HTT" \
        "$HG002_BAM" \
        "chr4" 3070000 3080000 \
        "Huntington disease locus (HTT CAG repeat, normal in HG002)"
fi

# ── Test 3: HG002 — FMR1 locus (Fragile X, normal allele) ──────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" || "$TEST_FILTER" == "fmr1" ]]; then
    run_test "HG002_FMR1" \
        "$HG002_BAM" \
        "chrX" 147907000 147917000 \
        "Fragile X syndrome locus (FMR1 CGG repeat, normal in HG002)"
fi

# ── Test 4: HG002 — FXN locus (Friedreich ataxia, normal allele) ───────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" || "$TEST_FILTER" == "fxn" ]]; then
    run_test "HG002_FXN" \
        "$HG002_BAM" \
        "chr9" 69032000 69042000 \
        "Friedreich ataxia locus (FXN GAA repeat, normal in HG002)"
fi

# ── Test 5: HG002 — DMPK locus (Myotonic dystrophy 1) ──────────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" || "$TEST_FILTER" == "dmpk" ]]; then
    run_test "HG002_DMPK" \
        "$HG002_BAM" \
        "chr19" 46268000 46278000 \
        "Myotonic dystrophy 1 locus (DMPK CTG repeat, normal in HG002)"
fi

# ── Test 6: HG002 — RFC1 locus (CANVAS) ────────────────────────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" || "$TEST_FILTER" == "rfc1" ]]; then
    run_test "HG002_RFC1" \
        "$HG002_BAM" \
        "chr9" 27568000 27578000 \
        "CANVAS locus (RFC1 AAGGG repeat, normal in HG002)"
fi

# ── Summary ─────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All tests complete. Results in: ${OUTPUT_BASE}/${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_BASE}/*/predictions_summary.json | python3 -m json.tool"
echo ""
echo "Quick one-liner to run a single test:"
echo "  uv run main.py --model-path output/str_model.joblib \\"
echo "    --predict-bam <BAM_FILE_OR_URL> \\"
echo "    --predict-chr chr22 --predict-start 45790000 --predict-end 45800000 \\"
echo "    --output-dir output/my_test"
