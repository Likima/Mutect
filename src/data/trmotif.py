"""
TRMotifAnnotator-inspired motif decomposition for tandem repeat analysis.

Implements single-base-resolution motif detection that identifies canonical
and non-canonical (variant) motifs within repeat sequences, handling:
- Interrupted repeats (e.g. CAGCAGCAACAGCAG)
- Motif rotations  (e.g. CAG == AGC == GCA for non-coding repeats)
- Multi-motif loci
- IUPAC degenerate base codes

Based on the approach described in:
    https://github.com/wf-TRs/TRMotifAnnotator

Also incorporates the STRling k-mer counting approach:
    Dashnow et al., Genome Biology (2022) doi:10.1186/s13059-022-02826-4

References for feature engineering:
    - RExPRT (Genome Biology 2024): genomic architecture features
    - HMMSTR (NAR 2025): HMM-based repeat copy counting
    - Pytrf (BMC Bioinformatics 2025): exact/approximate TR detection
"""

import logging
import re
from collections import Counter
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# IUPAC degenerate base expansion
IUPAC_MAP = {
    "A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"],
    "R": ["A", "G"], "Y": ["C", "T"], "S": ["G", "C"],
    "W": ["A", "T"], "K": ["G", "T"], "M": ["A", "C"],
    "B": ["C", "G", "T"], "D": ["A", "G", "T"],
    "H": ["A", "C", "T"], "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}


def expand_iupac(motif: str) -> List[str]:
    """Expand an IUPAC motif into all concrete sequences."""
    possibilities = [IUPAC_MAP.get(base.upper(), [base.upper()]) for base in motif]
    return ["".join(combo) for combo in product(*possibilities)]


def canonical_rotation(motif: str) -> str:
    """Return the lexicographically smallest rotation of a motif."""
    motif = motif.upper()
    rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
    return min(rotations)


def motifs_are_rotations(m1: str, m2: str) -> bool:
    """Check if two motifs are rotational equivalents."""
    if len(m1) != len(m2):
        return False
    return canonical_rotation(m1) == canonical_rotation(m2)


class TRMotifDecomposer:
    """Decompose a repeat sequence into its constituent motifs at single-base resolution.

    Inspired by TRMotifAnnotator, this identifies both canonical (expected)
    and non-canonical (variant) repeat units, handling interruptions and
    substitutions within repeats.
    """

    def __init__(
        self,
        min_unit_length: int = 1,
        max_unit_length: int = 6,
        handle_rotations: bool = True,
    ):
        self.min_unit_length = min_unit_length
        self.max_unit_length = max_unit_length
        self.handle_rotations = handle_rotations

    def decompose(
        self,
        sequence: str,
        canonical_motifs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decompose a sequence into motif annotations.

        Args:
            sequence: DNA sequence to decompose.
            canonical_motifs: Expected motif(s). If provided, the sequence
                is scanned for exact and variant copies of these motifs.

        Returns:
            Dictionary with decomposition results and derived features.
        """
        if not sequence or len(sequence) < 2:
            return self._empty_result()

        sequence = sequence.upper()

        # Step 1: Find the dominant motif (or use provided canonical motifs)
        if canonical_motifs:
            motifs_to_check = []
            for m in canonical_motifs:
                motifs_to_check.extend(expand_iupac(m))
        else:
            motifs_to_check = self._discover_motifs(sequence)

        if not motifs_to_check:
            return self._empty_result()

        # Step 2: Score each candidate motif
        # Use a composite score that rewards longer motifs with high coverage.
        # Pure coverage comparison lets mononucleotides win since every base
        # matches "T" with 1 mismatch tolerance.  The score is:
        #   coverage * (1 + log2(motif_length)) * canonical_fraction
        # This means ATTCT at 80% coverage with 90% canonical beats T at 100%.
        best = None
        best_score = -1.0
        for motif in motifs_to_check:
            result = self._annotate_with_motif(sequence, motif)
            mlen = max(result.get("motif_length", 1), 1)
            canon_frac = max(result.get("canonical_fraction", 0.0), 0.01)
            length_bonus = 1.0 + np.log2(mlen)
            score = result["coverage"] * length_bonus * canon_frac
            if score > best_score:
                best_score = score
                best = result

        if best is None or best["coverage"] < 0.1:
            return self._empty_result()

        return best

    def extract_features(
        self,
        sequence: str,
        canonical_motifs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Extract ML features from motif decomposition.

        Returns a flat dictionary of numeric features suitable for model input.
        """
        result = self.decompose(sequence, canonical_motifs)

        return {
            "trmotif_canonical_copies": float(result.get("canonical_copies", 0)),
            "trmotif_variant_copies": float(result.get("variant_copies", 0)),
            "trmotif_total_copies": float(result.get("total_copies", 0)),
            "trmotif_canonical_fraction": float(result.get("canonical_fraction", 0.0)),
            "trmotif_coverage": float(result.get("coverage", 0.0)),
            "trmotif_interruption_count": float(result.get("interruption_count", 0)),
            "trmotif_motif_length": float(result.get("motif_length", 0)),
            "trmotif_distinct_variants": float(result.get("distinct_variants", 0)),
            "trmotif_purity": float(result.get("purity", 0.0)),
        }

    # ------------------------------------------------------------------
    # Motif discovery
    # ------------------------------------------------------------------

    def _discover_motifs(self, sequence: str) -> List[str]:
        """Discover candidate repeat motifs in a sequence using k-mer frequency."""
        candidates = []

        for k in range(self.min_unit_length, self.max_unit_length + 1):
            if k > len(sequence) // 2:
                break

            # Non-overlapping k-mer counts (STRling approach)
            kmer_counts: Counter = Counter()
            for i in range(0, len(sequence) - k + 1, k):
                kmer = sequence[i : i + k]
                if "N" not in kmer:
                    norm = canonical_rotation(kmer) if self.handle_rotations else kmer
                    kmer_counts[norm] += 1

            # A motif is a candidate if it appears at least twice
            for kmer, count in kmer_counts.most_common(3):
                if count >= 2:
                    candidates.append(kmer)

        return candidates

    # ------------------------------------------------------------------
    # Motif annotation (single-base resolution)
    # ------------------------------------------------------------------

    def _annotate_with_motif(self, sequence: str, motif: str) -> Dict[str, Any]:
        """Annotate sequence with a single motif, counting canonical and variant copies."""
        motif_len = len(motif)
        seq_len = len(sequence)

        if motif_len == 0 or seq_len < motif_len:
            return self._empty_result()

        # Generate all rotations of the motif for matching
        if self.handle_rotations:
            rotations = set(motif[i:] + motif[:i] for i in range(motif_len))
        else:
            rotations = {motif}

        # Scan the sequence in non-overlapping windows
        canonical_copies = 0
        variant_copies = 0
        interruption_count = 0
        prev_was_match = False
        variant_set = set()
        matched_bases = 0

        pos = 0
        while pos + motif_len <= seq_len:
            window = sequence[pos : pos + motif_len]

            if window in rotations:
                # Exact canonical match
                canonical_copies += 1
                matched_bases += motif_len
                if not prev_was_match and pos > 0:
                    interruption_count += 1 if variant_copies > 0 else 0
                prev_was_match = True
                pos += motif_len
            else:
                # Check for variant copy (allow substitutions proportional to motif length)
                # k=1: 0 mismatches (no variants for mononucleotides)
                # k=2: 1 mismatch
                # k=3-6: 1 mismatch
                # k>6: up to motif_len//3
                mismatches = sum(1 for a, b in zip(window, motif) if a != b)
                max_mismatches = 0 if motif_len == 1 else max(1, motif_len // 3)
                if mismatches <= max_mismatches:
                    variant_copies += 1
                    variant_set.add(window)
                    matched_bases += motif_len
                    prev_was_match = True
                    pos += motif_len
                else:
                    if prev_was_match:
                        interruption_count += 1
                    prev_was_match = False
                    pos += 1  # slide by 1 to find next match

        total_copies = canonical_copies + variant_copies
        coverage = matched_bases / seq_len if seq_len > 0 else 0.0
        canonical_fraction = (
            canonical_copies / total_copies if total_copies > 0 else 0.0
        )
        purity = canonical_copies * motif_len / seq_len if seq_len > 0 else 0.0

        return {
            "motif": motif,
            "motif_length": motif_len,
            "canonical_copies": canonical_copies,
            "variant_copies": variant_copies,
            "total_copies": total_copies,
            "canonical_fraction": canonical_fraction,
            "coverage": coverage,
            "purity": purity,
            "interruption_count": interruption_count,
            "distinct_variants": len(variant_set),
            "variant_motifs": list(variant_set),
            "matched_bases": matched_bases,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "motif": "",
            "motif_length": 0,
            "canonical_copies": 0,
            "variant_copies": 0,
            "total_copies": 0,
            "canonical_fraction": 0.0,
            "coverage": 0.0,
            "purity": 0.0,
            "interruption_count": 0,
            "distinct_variants": 0,
            "variant_motifs": [],
            "matched_bases": 0,
        }


class STRlingKmerCounter:
    """STRling-inspired non-overlapping k-mer counting for STR detection.

    Uses the approach from Dashnow et al. (2022): counts non-overlapping k-mers
    in a read, stores minimal rotation for each, and determines the dominant
    repeat motif and its proportion.
    """

    def __init__(self, k_range: Tuple[int, int] = (2, 6)):
        self.k_min, self.k_max = k_range

    def count_features(self, sequence: str) -> Dict[str, float]:
        """Extract STRling-style k-mer features from a sequence.

        Returns:
            Dictionary of numeric features.
        """
        if not sequence or len(sequence) < 4:
            return {
                "strling_dominant_kmer_proportion": 0.0,
                "strling_dominant_kmer_count": 0.0,
                "strling_dominant_kmer_length": 0.0,
                "strling_total_repeat_kmers": 0.0,
                "strling_kmer_entropy": 0.0,
            }

        sequence = sequence.upper()
        best_proportion = 0.0
        best_count = 0
        best_k = 0
        best_entropy = 0.0

        for k in range(self.k_min, min(self.k_max + 1, len(sequence) // 2 + 1)):
            counts: Counter = Counter()
            n_windows = 0

            for i in range(0, len(sequence) - k + 1, k):
                kmer = sequence[i : i + k]
                if "N" not in kmer:
                    norm = canonical_rotation(kmer)
                    counts[norm] += 1
                    n_windows += 1

            if n_windows == 0:
                continue

            # Dominant k-mer proportion
            top_kmer, top_count = counts.most_common(1)[0]
            proportion = top_count * k / len(sequence)

            # Shannon entropy of k-mer distribution
            probs = np.array([c / n_windows for c in counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-12))

            if proportion > best_proportion:
                best_proportion = proportion
                best_count = top_count
                best_k = k
                best_entropy = entropy

        return {
            "strling_dominant_kmer_proportion": best_proportion,
            "strling_dominant_kmer_count": float(best_count),
            "strling_dominant_kmer_length": float(best_k),
            "strling_total_repeat_kmers": float(best_count),
            "strling_kmer_entropy": best_entropy,
        }
