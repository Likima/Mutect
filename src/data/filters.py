"""
Improved filtering with better separation of concerns and configurability.
"""
from typing import Any, List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterStrategy(Enum):
    """Strategies for handling variants with unknown values."""
    INCLUDE_UNKNOWN = "include"
    EXCLUDE_UNKNOWN = "exclude"
    ONLY_UNKNOWN = "only"


@dataclass
class FilterResult:
    """Result of a filtering operation."""
    kept: List[Dict]
    filtered_out: List[Dict]
    total: int
    kept_count: int
    filtered_count: int
    filter_name: str
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"{self.filter_name}: "
            f"kept={self.kept_count}, "
            f"filtered={self.filtered_count}, "
            f"total={self.total}"
        )


class VariantLengthCalculator:
    """Calculate variant length from various coordinate formats."""
    
    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        try:
            # Remove commas and whitespace
            clean_value = str(value).replace(",", "").strip()
            return int(float(clean_value))
        except (ValueError, TypeError, AttributeError):
            return None
    
    @staticmethod
    def _extract_coordinates(obj: Dict) -> tuple[Optional[int], Optional[int]]:
        """Extract start/end coordinates from an object."""
        start_keys = ("chr_start", "start", "position")
        end_keys = ("chr_end", "end", "stop", "chr_stop")
        
        start = end = None
        
        for key in start_keys:
            if key in obj:
                start = VariantLengthCalculator._coerce_int(obj.get(key))
                if start is not None:
                    break
        
        for key in end_keys:
            if key in obj:
                end = VariantLengthCalculator._coerce_int(obj.get(key))
                if end is not None:
                    break
        
        return start, end
    
    @staticmethod
    def _select_best_placement(placements: Any) -> Optional[Dict]:
        """Select best placement (prefer GRCh38 > GRCh37)."""
        if not placements:
            return None
        
        if isinstance(placements, dict):
            placements = [placements]
        elif not isinstance(placements, list):
            return None
        
        def rank(p):
            assembly = (p.get("assembly") or "").upper()
            if "GRCH38" in assembly or "HG38" in assembly:
                return 0
            if "GRCH37" in assembly or "HG19" in assembly:
                return 1
            return 2
        
        return sorted(placements, key=rank)[0]
    
    @classmethod
    def calculate_length(cls, variant: Dict) -> Optional[int]:
        """Calculate variant length from dbVar variant record.
        
        Tries multiple strategies:
        1. dbVar placement coordinates
        2. Top-level start/end fields
        3. Explicit length fields (svlen, length, size, etc.)
        
        Returns:
            Variant length in base pairs, or None if undeterminable
        """
        # Strategy 1: Placement coordinates
        placements = variant.get("dbvarplacementlist")
        best_placement = cls._select_best_placement(placements)
        
        if best_placement:
            start, end = cls._extract_coordinates(best_placement)
            if start is not None and end is not None:
                return abs(end - start) + 1
        
        # Strategy 2: Top-level coordinates
        start, end = cls._extract_coordinates(variant)
        if start is not None and end is not None:
            return abs(end - start) + 1
        
        # Strategy 3: Explicit length fields
        length_keys = ("svlen", "length", "variant_length", "size", "ins_length")
        for key in length_keys:
            length = cls._coerce_int(variant.get(key))
            if length is not None:
                return abs(length)
        
        return None


class VariantFilter:
    """Filter variants based on various criteria."""
    
    def __init__(self, length_calculator: Optional[VariantLengthCalculator] = None):
        """Initialize filter.
        
        Args:
            length_calculator: Custom length calculator (uses default if None)
        """
        self.length_calculator = length_calculator or VariantLengthCalculator()
    
    def filter_by_length(
        self,
        variants: List[Dict],
        min_bp: int = 50,
        max_bp: Optional[int] = None,
        unknown_strategy: FilterStrategy = FilterStrategy.EXCLUDE_UNKNOWN
    ) -> FilterResult:
        """Filter variants by length.
        
        Args:
            variants: List of variant dictionaries
            min_bp: Minimum length in base pairs
            max_bp: Maximum length in base pairs (None = no maximum)
            unknown_strategy: How to handle variants with unknown length
            
        Returns:
            FilterResult with kept and filtered variants
        """
        kept = []
        filtered_out = []
        
        for variant in variants:
            length = self.length_calculator.calculate_length(variant)
            
            if length is None:
                # Handle unknown length based on strategy
                if unknown_strategy == FilterStrategy.INCLUDE_UNKNOWN:
                    kept.append(variant)
                elif unknown_strategy == FilterStrategy.ONLY_UNKNOWN:
                    kept.append(variant)
                else:  # EXCLUDE_UNKNOWN
                    filtered_out.append(variant)
            elif unknown_strategy == FilterStrategy.ONLY_UNKNOWN:
                filtered_out.append(variant)
            else:
                # Check length bounds
                passes_min = length >= min_bp
                passes_max = max_bp is None or length <= max_bp
                
                if passes_min and passes_max:
                    kept.append(variant)
                else:
                    filtered_out.append(variant)
        
        return FilterResult(
            kept=kept,
            filtered_out=filtered_out,
            total=len(variants),
            kept_count=len(kept),
            filtered_count=len(filtered_out),
            filter_name=f"length_{min_bp}bp"
        )
    
    def filter_by_custom(
        self,
        variants: List[Dict],
        predicate: Callable[[Dict], bool],
        filter_name: str = "custom_filter"
    ) -> FilterResult:
        """Filter variants using custom predicate function.
        
        Args:
            variants: List of variant dictionaries
            predicate: Function that returns True to keep variant
            filter_name: Name for the filter (for logging)
            
        Returns:
            FilterResult with kept and filtered variants
        """
        kept = []
        filtered_out = []
        
        for variant in variants:
            try:
                if predicate(variant):
                    kept.append(variant)
                else:
                    filtered_out.append(variant)
            except Exception as e:
                logger.warning(f"Error in custom filter: {e}")
                filtered_out.append(variant)
        
        return FilterResult(
            kept=kept,
            filtered_out=filtered_out,
            total=len(variants),
            kept_count=len(kept),
            filtered_count=len(filtered_out),
            filter_name=filter_name
        )


class VariantSummarizer:
    """Generate summary statistics for variants."""
    
    @staticmethod
    def summarize_clinical_significance(variants: List[Dict]) -> Dict[str, int]:
        """Count variants by clinical significance.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Dictionary mapping significance -> count
        """
        counts = {}
        
        for variant in variants:
            significance = variant.get("clinical_significance", "N/A")
            
            if isinstance(significance, list):
                for item in significance:
                    sig_str = str(item)
                    counts[sig_str] = counts.get(sig_str, 0) + 1
            else:
                sig_str = str(significance)
                counts[sig_str] = counts.get(sig_str, 0) + 1
        
        return counts
    
    @staticmethod
    def print_summary(variants: List[Dict]):
        """Print clinical significance summary.
        
        Args:
            variants: List of variant dictionaries
        """
        counts = VariantSummarizer.summarize_clinical_significance(variants)
        
        print("\nClinical Significance Summary:")
        for significance, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {significance}: {count}")


# Backward compatibility functions
def summarize_variants(variants: List[Dict]):
    """Legacy function for backward compatibility."""
    VariantSummarizer.print_summary(variants)


def filter_str_by_min_length(
    variants: List[Dict], 
    min_bp: int = 50, 
    include_unknown: bool = False
) -> List[Dict]:
    """Legacy function for backward compatibility."""
    strategy = FilterStrategy.INCLUDE_UNKNOWN if include_unknown else FilterStrategy.EXCLUDE_UNKNOWN
    
    filter_obj = VariantFilter()
    result = filter_obj.filter_by_length(variants, min_bp=min_bp, unknown_strategy=strategy)
    
    logger.info(result.summary())
    
    return result.kept