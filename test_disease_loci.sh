#!/usr/bin/env bash
# test_disease_loci.sh — Test the STR pipeline against known pathogenic TR loci
#
# Tests confirmed pathogenic repeat expansions from the 1000 Genomes ONT dataset
# and normal alleles from GIAB HG002 as negative controls.
#
# True positives (confirmed by REDatlas / Gustafson et al. 2024):
#   HG01122  ATXN10  ATTCT x1043  full_mutation    (SCA10)
#   HG02252  ATXN10  ATTCT x955   full_mutation    (SCA10)
#   HG02345  ATXN10  ATTCT x321   reduced_penetrance (SCA10)
#   HG00105  RFC1    AAAAG x682   full_mutation    (CANVAS)
#   HG01122  RFC1    AAAAG x656   full_mutation    (CANVAS)
#   HG00110  FGF14   GAA   x251   reduced_penetrance (SCA27B)
#   HG01501  FGF14   GAA   x275   reduced_penetrance (SCA27B)
#
# True negatives (HG002 is a healthy GIAB reference individual):
#   HG002    HTT     CAG   ~17    normal
#   HG002    FMR1    CGG   ~30    normal
#
# Usage:
#   ./test_disease_loci.sh              # Run all tests
#   ./test_disease_loci.sh atxn10       # ATXN10 tests only
#   ./test_disease_loci.sh rfc1         # RFC1 tests only
#   ./test_disease_loci.sh fgf14        # FGF14 tests only
#   ./test_disease_loci.sh hg002        # HG002 negative controls only
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

# S3 base URL for 1KGP ONT BAMs
ONT_BASE="https://1000g-ont.s3.amazonaws.com/ALIGNMENT_AND_ASSEMBLY_DATA/FIRST_100/NAPU_PIPELINE/HG38"

# GIAB HG002 PacBio BAM
HG002_BAM="https://downloads.pacbcloud.com/public/dataset/HG002-CpG-methylation-202202/HG002.GRCh38.haplotagged.bam"

ont_bam() {
    local sample="$1"
    local suffix="$2"
    echo "${ONT_BASE}/${sample}-ONT-hg38-R9-LSK110-guppy-sup-${suffix}/${sample}-ONT-hg38-R9-LSK110-guppy-sup-${suffix}.PMDV_FINAL.haplotagged.bam"
}

# Resolve BAM: prefer local slice, fall back to remote
resolve_bam() {
    local local_bam="$1"
    local remote_bam="$2"
    if [ -f "$local_bam" ]; then
        echo "$local_bam"
    else
        echo "$remote_bam"
    fi
}

# Download a BAM region slice if samtools is available and local BAM doesn't exist
slice_bam() {
    local remote_bam="$1"
    local local_bam="$2"
    local region="$3"
    if [ -f "$local_bam" ]; then
        return 0
    fi
    if ! command -v samtools &> /dev/null; then
        echo -e "${YELLOW}  samtools not found — using remote BAM (slower, may fail)${NC}"
        return 1
    fi
    echo -e "${YELLOW}  Downloading BAM slice: ${local_bam}${NC}"
    samtools view -b "$remote_bam" "$region" > "$local_bam" 2>/dev/null && \
    samtools index "$local_bam" 2>/dev/null && \
    echo -e "${GREEN}  Downloaded and indexed: ${local_bam}${NC}" || \
    { echo -e "${RED}  Failed to slice BAM${NC}"; rm -f "$local_bam"; return 1; }
}

# ── Helper ──────────────────────────────────────────────────────────────────

run_test() {
    local name="$1"
    local bam="$2"
    local chr="$3"
    local start="$4"
    local end="$5"
    local disease="$6"
    local expected="$7"  # "positive" or "negative"
    local outdir="${OUTPUT_BASE}/${name}"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  TEST: ${name}${NC}"
    echo -e "${CYAN}  Disease: ${disease}${NC}"
    echo -e "${CYAN}  Region: ${chr}:${start}-${end}${NC}"
    echo -e "${CYAN}  Expected: ${expected}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    mkdir -p "$outdir"

    if [ -f "$MODEL_PATH" ]; then
        uv run main.py \
            --model-path "$MODEL_PATH" \
            --predict-bam "$bam" \
            --predict-chr "$chr" \
            --predict-start "$start" \
            --predict-end "$end" \
            --output-dir "$outdir" \
            2>&1 | tail -20
    else
        uv run main.py \
            --str-data "$STR_DATA" \
            --normal-data "$NORMAL_DATA" \
            --train \
            --predict-bam "$bam" \
            --predict-chr "$chr" \
            --predict-start "$start" \
            --predict-end "$end" \
            --output-dir "$outdir" \
            2>&1 | tail -20
    fi

    # Show results
    if [ -f "${outdir}/predictions_summary.json" ]; then
        echo ""
        python -c "
import json
with open('${outdir}/predictions_summary.json') as f:
    s = json.load(f)
strs = s['predicted_strs']
total = s['total_sequences']
expected = '${expected}'
icon = '✓' if (expected == 'positive' and strs > 0) or (expected == 'negative' and strs == 0) else '✗'
status = 'PASS' if icon == '✓' else 'FAIL'
print(f'  {icon} {status}: {strs}/{total} STRs detected (expected: {expected})')
ca = s.get('clinical_annotations', {})
if ca.get('total_disease_matches', 0) > 0:
    for d, c in ca.get('disease_matches', {}).items():
        print(f'    Disease: {d} ({c} match)')
for p in s.get('top_20_str_predictions', [])[:3]:
    motif = p.get('repeat_motif', '?')
    prob = p.get('probability', 0)
    rc = p.get('repeat_count', 0)
    ac = p.get('redatlas_allele_class', '')
    print(f'    Motif={motif} x{rc}, prob={prob:.4f}, class={ac}')
" 2>/dev/null || true
    fi
}

# ── Ensure training data exists ─────────────────────────────────────────────

if [ ! -f "$STR_DATA" ] || [ ! -f "$NORMAL_DATA" ]; then
    echo -e "${RED}Training data not found. Run the default pipeline first.${NC}"
    exit 1
fi

# ── Train model once if not present ─────────────────────────────────────────

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Training model (one-time)...${NC}"
    uv run main.py \
        --str-data "$STR_DATA" \
        --normal-data "$NORMAL_DATA" \
        --train --output-dir output
fi

TEST_FILTER="${1:-all}"
mkdir -p "$OUTPUT_BASE"

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  STR Disease Loci Test Suite${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

# ═══════════════════════════════════════════════════════════════════════════
# TRUE POSITIVES — confirmed pathogenic expansions
# ═══════════════════════════════════════════════════════════════════════════

# ── ATXN10: Spinocerebellar ataxia 10 (chr22:45795355, ATTCT) ────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "atxn10" ]]; then
    # HG01122: ATTCT x1043, full_mutation
    slice_bam "$(ont_bam HG01122 5mC)" "HG01122_ATXN10.bam" "chr22:45790000-45800000"
    run_test "HG01122_ATXN10" "$(resolve_bam HG01122_ATXN10.bam "$(ont_bam HG01122 5mC)")" \
        "chr22" 45790000 45800000 \
        "SCA10: HG01122 ATTCT x1043 full_mutation" "positive"

    # HG02252: ATTCT x955 (full_mutation) + x511 (reduced_penetrance)
    slice_bam "$(ont_bam HG02252 5mC)" "HG02252_ATXN10.bam" "chr22:45790000-45800000"
    run_test "HG02252_ATXN10" "$(resolve_bam HG02252_ATXN10.bam "$(ont_bam HG02252 5mC)")" \
        "chr22" 45790000 45800000 \
        "SCA10: HG02252 ATTCT x955 full_mutation" "positive"

    # HG02345: ATTCT x321, reduced_penetrance
    slice_bam "$(ont_bam HG02345 5mC)" "HG02345_ATXN10.bam" "chr22:45790000-45800000"
    run_test "HG02345_ATXN10" "$(resolve_bam HG02345_ATXN10.bam "$(ont_bam HG02345 5mC)")" \
        "chr22" 45790000 45800000 \
        "SCA10: HG02345 ATTCT x321 reduced_penetrance" "positive"
fi

# ── RFC1: CANVAS (chr4:39348425, AAAAG) ─────────────────────────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "rfc1" ]]; then
    # HG00105: AAAAG x682, full_mutation
    slice_bam "$(ont_bam HG00105 5mC)" "HG00105_RFC1.bam" "chr4:39343000-39353000"
    run_test "HG00105_RFC1" "$(resolve_bam HG00105_RFC1.bam "$(ont_bam HG00105 5mC)")" \
        "chr4" 39343000 39353000 \
        "CANVAS: HG00105 AAAAG x682 full_mutation" "positive"

    # HG01122: AAAAG x656, full_mutation (same sample as ATXN10!)
    slice_bam "$(ont_bam HG01122 5mC)" "HG01122_RFC1.bam" "chr4:39343000-39353000"
    run_test "HG01122_RFC1" "$(resolve_bam HG01122_RFC1.bam "$(ont_bam HG01122 5mC)")" \
        "chr4" 39343000 39353000 \
        "CANVAS: HG01122 AAAAG x656 full_mutation" "positive"
fi

# ── FGF14: Spinocerebellar ataxia 27B (chr13:102161577, GAA) ────────────

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "fgf14" ]]; then
    # HG00110: GAA x251, reduced_penetrance
    slice_bam "$(ont_bam HG00110 5mC)" "HG00110_FGF14.bam" "chr13:102156000-102167000"
    run_test "HG00110_FGF14" "$(resolve_bam HG00110_FGF14.bam "$(ont_bam HG00110 5mC)")" \
        "chr13" 102156000 102167000 \
        "SCA27B: HG00110 GAA x251 reduced_penetrance" "positive"

    # HG01501: GAA x275, reduced_penetrance
    slice_bam "$(ont_bam HG01501 5hmc_5mc_cg)" "HG01501_FGF14.bam" "chr13:102156000-102167000"
    run_test "HG01501_FGF14" "$(resolve_bam HG01501_FGF14.bam "$(ont_bam HG01501 5hmc_5mc_cg)")" \
        "chr13" 102156000 102167000 \
        "SCA27B: HG01501 GAA x275 reduced_penetrance" "positive"
fi

# ═══════════════════════════════════════════════════════════════════════════
# TRUE NEGATIVES — HG002 healthy reference, normal alleles
# ═══════════════════════════════════════════════════════════════════════════

if [[ "$TEST_FILTER" == "all" || "$TEST_FILTER" == "hg002" ]]; then
    # HTT: Huntington disease (normal ~17 CAG)
    run_test "HG002_HTT" "$HG002_BAM" \
        "chr4" 3070000 3080000 \
        "Huntington: HG002 normal allele" "negative"

    # FMR1: Fragile X (normal ~30 CGG)
    run_test "HG002_FMR1" "$HG002_BAM" \
        "chrX" 147907000 147917000 \
        "Fragile X: HG002 normal allele" "negative"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All tests complete. Results in: ${OUTPUT_BASE}/${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Quick summary:"
python -c "
import json, os, glob
results = []
for p in sorted(glob.glob('${OUTPUT_BASE}/*/predictions_summary.json')):
    name = os.path.basename(os.path.dirname(p))
    with open(p) as f:
        s = json.load(f)
    strs = s['predicted_strs']
    total = s['total_sequences']
    diseases = list(s.get('clinical_annotations', {}).get('disease_matches', {}).keys())
    disease_str = diseases[0] if diseases else '-'
    results.append((name, total, strs, disease_str))

print(f'  {\"Test\":<30s} {\"Seqs\":>5s} {\"STRs\":>5s}  Disease')
print(f'  {\"-\"*30} {\"-\"*5} {\"-\"*5}  {\"-\"*30}')
for name, total, strs, disease in results:
    print(f'  {name:<30s} {total:>5d} {strs:>5d}  {disease}')
" 2>/dev/null || true
