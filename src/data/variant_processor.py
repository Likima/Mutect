"""
Improved variant processing with better type safety and error handling.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GenomeAssembly(Enum):
    """Supported genome assemblies."""
    GRCH38 = "GRCH38"
    GRCH37 = "GRCH37"
    HG38 = "HG38"
    HG19 = "HG19"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProcessedVariant:
    """Structured representation of a processed variant."""
    uid: str
    gene: str
    title: str
    chromosome: str
    start: str
    end: str
    clinical_significance: Any
    review_status: str = "N/A"
    condition: str = "N/A"
    consequence: str = "N/A"
    assembly: str = "N/A"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def is_valid_position(self) -> bool:
        """Check if variant has valid genomic coordinates."""
        return self.start != "?" and self.end != "?" and self.chromosome != "N/A"
    
    @property
    def length(self) -> Optional[int]:
        """Calculate variant length if positions are valid."""
        if not self.is_valid_position:
            return None
        try:
            return int(self.end) - int(self.start) + 1
        except (ValueError, TypeError):
            return None


class VariantProcessor:
    """Process raw dbVar variants into structured format."""
    
    ASSEMBLY_PRIORITY = [
        GenomeAssembly.GRCH38,
        GenomeAssembly.GRCH37,
        GenomeAssembly.HG38,
        GenomeAssembly.HG19,
    ]
    
    @staticmethod
    def _normalize_assembly(assembly_str: str) -> GenomeAssembly:
        """Normalize assembly string to GenomeAssembly enum."""
        assembly_upper = assembly_str.upper().replace("_", "").replace("-", "")
        
        for assembly in GenomeAssembly:
            if assembly.value in assembly_upper:
                return assembly
        
        return GenomeAssembly.UNKNOWN
    
    @staticmethod
    def _rank_placement(placement: Dict) -> int:
        """Rank placement by assembly preference."""
        assembly = VariantProcessor._normalize_assembly(
            placement.get("assembly", "")
        )
        
        try:
            return VariantProcessor.ASSEMBLY_PRIORITY.index(assembly)
        except ValueError:
            return len(VariantProcessor.ASSEMBLY_PRIORITY)
    
    @staticmethod
    def _extract_best_placement(placements: Any) -> Optional[Dict]:
        """Extract best genomic placement from dbVar placements.
        
        Prefers GRCh38 > GRCh37 > others.
        """
        if not placements:
            return None
        
        # Normalize to list
        if isinstance(placements, dict):
            placements = [placements]
        elif not isinstance(placements, list):
            return None
        
        if not placements:
            return None
        
        # Sort by assembly preference
        sorted_placements = sorted(placements, key=VariantProcessor._rank_placement)
        return sorted_placements[0]
    
    @staticmethod
    def _extract_gene(variant: Dict) -> str:
        """Extract gene name from variant record."""
        gene_list = variant.get("dbvargenelist") or variant.get("genes") or []
        
        if not isinstance(gene_list, list) or not gene_list:
            return "N/A"
        
        first_gene = gene_list[0]
        
        if isinstance(first_gene, dict):
            # Try multiple possible field names
            for field in ["name", "symbol", "gene_symbol", "gene_name"]:
                gene_name = first_gene.get(field)
                if gene_name:
                    return str(gene_name)
        elif isinstance(first_gene, str):
            return first_gene
        
        return "N/A"
    
    @staticmethod
    def _safe_str(value: Any, default: str = "?") -> str:
        """Safely convert value to string."""
        if value is None:
            return default
        return str(value)
    
    @classmethod
    def process_variant(cls, variant: Dict) -> ProcessedVariant:
        """Process a single dbVar variant into structured format.
        
        Args:
            variant: Raw variant dictionary from dbVar
            
        Returns:
            ProcessedVariant object
        """
        uid = variant.get("uid", "N/A")
        title = variant.get("title", "N/A")
        gene = cls._extract_gene(variant)
        
        # Extract genomic coordinates
        chromosome = "N/A"
        start = "?"
        end = "?"
        assembly = "N/A"
        
        placements = variant.get("dbvarplacementlist")
        best_placement = cls._extract_best_placement(placements)
        
        if best_placement:
            chromosome = cls._safe_str(best_placement.get("chr"), "N/A")
            start = cls._safe_str(
                best_placement.get("chr_start") or 
                best_placement.get("start") or 
                best_placement.get("position")
            )
            end = cls._safe_str(
                best_placement.get("chr_end") or 
                best_placement.get("stop") or 
                best_placement.get("end")
            )
            assembly = cls._safe_str(best_placement.get("assembly"), "N/A")
        
        # Extract clinical information
        clinical_significance = variant.get("dbvarclinicalsignificancelist")
        if clinical_significance in (None, [], {}):
            clinical_significance = "N/A"
        
        return ProcessedVariant(
            uid=uid,
            gene=gene,
            title=title,
            chromosome=chromosome,
            start=start,
            end=end,
            clinical_significance=clinical_significance,
            review_status=variant.get("review_status", "N/A"),
            condition=variant.get("condition", "N/A"),
            consequence=variant.get("consequence", "N/A"),
            assembly=assembly
        )
    
    @classmethod
    def process_variants(cls, variants: List[Dict]) -> List[ProcessedVariant]:
        """Process multiple variants.
        
        Args:
            variants: List of raw variant dictionaries
            
        Returns:
            List of ProcessedVariant objects
        """
        processed = []
        errors = 0
        
        for i, variant in enumerate(variants):
            try:
                processed_variant = cls.process_variant(variant)
                processed.append(processed_variant)
            except Exception as e:
                errors += 1
                logger.error(f"Error processing variant {i}: {e}")
        
        if errors:
            logger.warning(f"Failed to process {errors}/{len(variants)} variants")
        
        return processed
    
    @classmethod
    def to_dict_list(cls, variants: List[ProcessedVariant]) -> List[Dict]:
        """Convert ProcessedVariant objects to dictionaries."""
        return [v.to_dict() for v in variants]


# Backward compatibility
def pass_through_variants(variants: List[Dict]) -> List[Dict]:
    """Legacy function for backward compatibility."""
    processor = VariantProcessor()
    processed = processor.process_variants(variants)
    return processor.to_dict_list(processed)