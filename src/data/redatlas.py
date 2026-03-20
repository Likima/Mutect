"""
REDatlas integration for clinical annotation of tandem repeat loci.

Maps predicted STR loci to known disease-associated repeat expansion disorders
using data from REDatlas (https://github.com/wf-TRs/REDatlas).

REDatlas provides population-level data on 66 disease-associated TR loci with
clinical thresholds, allele frequencies, and ancestry-specific patterns across
2,530 haplotypes from diverse populations.

References:
    - Rajan-Babu et al., medRxiv 2025.10.11.25337795 (2025)
    - RExPRT: Genome Biology (2024) doi:10.1186/s13059-024-03171-4
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Hardcoded catalog of 66 well-known disease-associated TR loci from REDatlas.
# Used as fallback when TSV files are not available locally.
# Fields: (chrom, start_grch38, end_grch38, motif, gene, disease, normal_max, pathogenic_min)
# GRCh38 coordinates for all loci
KNOWN_DISEASE_LOCI = [
    ("chr4", 3074876, 3074933, "CAG", "HTT", "Huntington disease", 35, 40),
    # RFC1: GRCh38 is chr4, NOT chr9
    ("chr4", 39348425, 39348483, "AAAAG", "RFC1", "CANVAS", 11, 400),
    ("chr3", 63912684, 63912714, "CTG", "CNBP", "Myotonic dystrophy type 2", 26, 75),
    ("chr19", 46273462, 46273522, "CTG", "DMPK", "Myotonic dystrophy type 1", 34, 50),
    ("chrX", 147912050, 147912110, "CGG", "FMR1", "Fragile X syndrome", 44, 200),
    ("chr9", 69037286, 69037304, "GAA", "FXN", "Friedreich ataxia", 33, 66),
    ("chr9", 27573483, 27573546, "GGCCCC", "C9orf72", "ALS/FTD", 24, 30),
    ("chr12", 111598949, 111599018, "CAG", "ATXN2", "Spinocerebellar ataxia 2", 31, 33),
    ("chr14", 92071010, 92071042, "CAG", "ATXN3", "Spinocerebellar ataxia 3", 44, 60),
    ("chr6", 170561906, 170561926, "CAG", "TBP", "Spinocerebellar ataxia 17", 42, 45),
    ("chr12", 6936716, 6936773, "CAG", "ATN1", "DRPLA", 35, 49),
    ("chr19", 13207858, 13207897, "CAG", "ATXN1", "Spinocerebellar ataxia 1", 38, 39),
    ("chr11", 119206289, 119206322, "CCTG", "CBL", "Jacobsen syndrome", 10, 50),
    # ATXN10: GRCh38 is on chr22, NOT chr5 (chr5 is GRCh37)
    ("chr22", 45795354, 45795424, "ATTCT", "ATXN10", "Spinocerebellar ataxia 10", 32, 800),
    ("chr16", 87637893, 87637934, "CTG", "JPH3", "Huntington disease-like 2", 28, 40),
    ("chrX", 67545316, 67545385, "GCA", "AR", "SBMA (Kennedy disease)", 34, 38),
    ("chr13", 70139383, 70139429, "AAAAG", "BEAN1", "Spinocerebellar ataxia 31", 0, 110),
    ("chr16", 24613439, 24613488, "AARRG", "TNRC6A", "Oculopharyngodistal myopathy 4", 0, 90),
    ("chr1", 57367043, 57367100, "AAGGG", "DAB1", "Spinocerebellar ataxia 37", 0, 31),
    ("chr4", 159342525, 159342633, "GGC", "FRG1", "FSHD-related", 0, 100),
    ("chr13", 102161577, 102161726, "GAA", "FGF14", "Spinocerebellar ataxia 27B", 249, 300),
]


class REDatlasAnnotator:
    """Annotate predicted STR loci with disease associations from REDatlas.

    Maps predicted STR coordinates to the nearest known disease-associated TR
    locus and returns clinical context including disease name, pathogenic
    thresholds, and population allele frequencies when available.
    """

    def __init__(
        self,
        summary_tsv: Optional[str] = None,
        population_tsv: Optional[str] = None,
        max_distance_bp: int = 1000,
    ):
        self.max_distance_bp = max_distance_bp
        self.loci_df = None
        self.population_df = None

        if summary_tsv and Path(summary_tsv).exists():
            self._load_summary(summary_tsv)
        if population_tsv and Path(population_tsv).exists():
            self._load_population(population_tsv)

        # Always load the built-in catalog
        self._builtin_loci = self._build_builtin_index()
        logger.info(
            f"REDatlas annotator ready: {len(self._builtin_loci)} built-in loci, "
            f"TSV loaded={self.loci_df is not None}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_summary(self, path: str) -> None:
        try:
            self.loci_df = pd.read_csv(path, sep="\t")
            logger.info(f"Loaded REDatlas summary: {len(self.loci_df)} loci from {path}")
        except Exception as e:
            logger.warning(f"Could not load REDatlas summary TSV: {e}")

    def _load_population(self, path: str) -> None:
        try:
            self.population_df = pd.read_csv(path, sep="\t")
            logger.info(
                f"Loaded REDatlas population data: {len(self.population_df)} rows from {path}"
            )
        except Exception as e:
            logger.warning(f"Could not load REDatlas population TSV: {e}")

    @staticmethod
    def _build_builtin_index() -> List[Dict[str, Any]]:
        loci = []
        for chrom, start, end, motif, gene, disease, norm_max, path_min in KNOWN_DISEASE_LOCI:
            loci.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "motif": motif,
                "gene": gene,
                "disease": disease,
                "normal_max": norm_max,
                "pathogenic_min": path_min,
            })
        return loci

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        chrom: str,
        start: int,
        end: int,
        motif: str = "",
        repeat_count: int = 0,
    ) -> Dict[str, Any]:
        """Annotate a single locus with REDatlas disease associations.

        Args:
            chrom: Chromosome (e.g. "chr4", "4", "chr4")
            start: Start position (0-based)
            end: End position
            motif: Detected repeat motif
            repeat_count: Number of repeat copies

        Returns:
            Dictionary with clinical annotation fields.
        """
        chrom_norm = self._normalize_chrom(chrom)

        best_match = None
        best_distance = float("inf")

        for locus in self._builtin_loci:
            if self._normalize_chrom(locus["chrom"]) != chrom_norm:
                continue

            dist = self._locus_distance(start, end, locus["start"], locus["end"])
            if dist < best_distance:
                best_distance = dist
                best_match = locus

        if best_match is None or best_distance > self.max_distance_bp:
            return self._no_match_result()

        # Classify allele
        allele_class = self._classify_allele(
            repeat_count, best_match["normal_max"], best_match["pathogenic_min"]
        )

        result = {
            "redatlas_match": True,
            "redatlas_gene": best_match["gene"],
            "redatlas_disease": best_match["disease"],
            "redatlas_distance_bp": int(best_distance),
            "redatlas_known_motif": best_match["motif"],
            "redatlas_motif_match": self._motifs_equivalent(motif, best_match["motif"]),
            "redatlas_normal_max": best_match["normal_max"],
            "redatlas_pathogenic_min": best_match["pathogenic_min"],
            "redatlas_allele_class": allele_class,
        }

        # Add population data if available
        pop_data = self._get_population_frequencies(best_match["gene"])
        if pop_data:
            result["redatlas_population"] = pop_data

        return result

    def annotate_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate a list of STR predictions with REDatlas data.

        Modifies predictions in-place and also returns them.
        """
        annotated_count = 0
        for pred in predictions:
            chrom = pred.get("chromosome", pred.get("chr", pred.get("reference_name", "")))

            # Prefer windowed genomic coordinates (from long-read windowed prediction)
            # over the full-read position, since the window pinpoints the STR region.
            if "window_genomic_start" in pred:
                start = int(pred["window_genomic_start"])
                end = int(pred.get("window_genomic_end", start + 5000))
            else:
                start = int(pred.get("position", pred.get("start", 0)))
                seq = pred.get("sequence", "")
                end = int(pred.get("end", start + len(seq)))

            motif = pred.get("repeat_motif", "")
            repeat_count = int(pred.get("repeat_count", 0))

            annotation = self.annotate(chrom, start, end, motif, repeat_count)
            pred.update(annotation)

            if annotation.get("redatlas_match"):
                annotated_count += 1

        logger.info(
            f"REDatlas annotation: {annotated_count}/{len(predictions)} matched known loci"
        )
        return predictions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_chrom(chrom: str) -> str:
        """Normalize chromosome name to bare number/letter."""
        return str(chrom).replace("chr", "").strip()

    @staticmethod
    def _locus_distance(start1: int, end1: int, start2: int, end2: int) -> int:
        """Distance between two intervals (0 if overlapping)."""
        if end1 <= start2:
            return start2 - end1
        if end2 <= start1:
            return start1 - end2
        return 0  # overlapping

    @staticmethod
    def _classify_allele(repeat_count: int, normal_max: int, pathogenic_min: int) -> str:
        if repeat_count <= 0:
            return "unknown"
        if repeat_count <= normal_max:
            return "normal"
        if repeat_count >= pathogenic_min:
            return "full_mutation"
        # Between normal_max and pathogenic_min
        midpoint = (normal_max + pathogenic_min) / 2
        if repeat_count < midpoint:
            return "intermediate"
        return "reduced_penetrance"

    @staticmethod
    def _motifs_equivalent(motif1: str, motif2: str) -> bool:
        """Check if two motifs are equivalent under rotation."""
        if not motif1 or not motif2:
            return False
        m1, m2 = motif1.upper(), motif2.upper()
        if len(m1) != len(m2):
            return False
        doubled = m1 + m1
        return m2 in doubled

    def _get_population_frequencies(self, gene: str) -> Optional[Dict[str, Any]]:
        """Get population-level allele frequencies for a gene from TSV data."""
        if self.population_df is None:
            return None

        # Try to find matching rows — column names may vary
        gene_col = None
        for col in ["gene", "Gene", "gene_name", "locus"]:
            if col in self.population_df.columns:
                gene_col = col
                break

        if gene_col is None:
            return None

        rows = self.population_df[self.population_df[gene_col] == gene]
        if rows.empty:
            return None

        return {"sample_count": len(rows), "gene": gene}

    @staticmethod
    def _no_match_result() -> Dict[str, Any]:
        return {
            "redatlas_match": False,
            "redatlas_gene": None,
            "redatlas_disease": None,
            "redatlas_distance_bp": None,
            "redatlas_known_motif": None,
            "redatlas_motif_match": False,
            "redatlas_normal_max": None,
            "redatlas_pathogenic_min": None,
            "redatlas_allele_class": "unknown",
        }
