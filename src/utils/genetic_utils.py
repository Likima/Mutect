"""
Genetic utilities for STR detection and deduplication.

This module provides functions for detecting and merging duplicate or adjacent
STR predictions that represent the same tandem repeat region.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate the overlap between two genomic intervals.
    
    Args:
        start1, end1: First interval (0-based, end-exclusive)
        start2, end2: Second interval (0-based, end-exclusive)
        
    Returns:
        Number of overlapping base pairs (0 if no overlap)
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def calculate_distance(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate the distance between two genomic intervals.
    
    Args:
        start1, end1: First interval (0-based, end-exclusive)
        start2, end2: Second interval (0-based, end-exclusive)
        
    Returns:
        Distance in base pairs (0 if overlapping, positive if gap exists)
    """
    if end1 <= start2:
        return start2 - end1  # Gap between intervals
    elif end2 <= start1:
        return start1 - end2  # Gap between intervals
    else:
        return 0  # Overlapping


def merge_sequences(seq1: str, pos1: int, seq2: str, pos2: int) -> Tuple[str, int, int]:
    """Merge two sequences that may overlap or be adjacent.
    
    Args:
        seq1: First sequence
        pos1: Start position of first sequence (0-based)
        seq2: Second sequence
        pos2: Start position of second sequence (0-based)
        
    Returns:
        Tuple of (merged_sequence, merged_start, merged_end)
    """
    end1 = pos1 + len(seq1)
    end2 = pos2 + len(seq2)
    
    # Determine order
    if pos1 <= pos2:
        first_seq, first_pos = seq1, pos1
        second_seq, second_pos = seq2, pos2
        first_end = end1
        second_end = end2
    else:
        first_seq, first_pos = seq2, pos2
        second_seq, second_pos = seq1, pos1
        first_end = end2
        second_end = end1
    
    # Check for overlap
    overlap = calculate_overlap(first_pos, first_end, second_pos, second_end)
    
    if overlap > 0:
        # Sequences overlap - merge by taking the overlapping region from first sequence
        # and appending the non-overlapping part of second sequence
        overlap_start_in_first = second_pos - first_pos
        overlap_end_in_first = overlap_start_in_first + overlap
        
        # Use the overlapping region from first sequence
        merged = first_seq[:overlap_end_in_first]
        
        # Append non-overlapping part of second sequence
        if second_end > first_end:
            # Second sequence extends beyond first
            non_overlap_start = overlap
            merged += second_seq[non_overlap_start:]
        # If second sequence is completely within first, just use first
        
    else:
        # No overlap - sequences are adjacent or have a gap
        gap = second_pos - first_end
        
        if gap == 0:
            # Adjacent - concatenate directly
            merged = first_seq + second_seq
        else:
            # Gap exists - fill with N's or concatenate (depending on preference)
            # For STR detection, we'll concatenate with gap filled by N's
            # This preserves the structure while indicating uncertainty
            gap_fill = 'N' * gap
            merged = first_seq + gap_fill + second_seq
    
    merged_start = min(first_pos, second_pos)
    merged_end = merged_start + len(merged)
    
    return merged, merged_start, merged_end


def detect_and_merge_duplicate_strs(
    predicted_strs: List[Dict[str, Any]],
    max_gap: int = 50,
    min_overlap: int = 5,
    require_same_chromosome: bool = True,
    reclassify_merged: bool = False,
    classifier: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Detect and merge duplicate or adjacent STR predictions.
    
    This function identifies STR predictions that:
    1. Overlap by at least min_overlap base pairs, OR
    2. Are within max_gap base pairs of each other
    
    And merges them into a single STR prediction.
    
    Args:
        predicted_strs: List of STR predictions, each containing:
            - 'sequence': DNA sequence
            - 'chromosome': Chromosome name
            - 'position': Start position (0-based) OR 'start' and 'end' positions
            - 'predicted_str': Boolean indicating STR prediction
            - 'str_probability': Probability of being STR
            - Other metadata fields
        max_gap: Maximum gap (in bp) between STRs to consider for merging
        min_overlap: Minimum overlap (in bp) to consider STRs as duplicates
        require_same_chromosome: If True, only merge STRs on same chromosome
        reclassify_merged: If True, re-run classifier on merged sequences
        classifier: STR_Classifier instance (required if reclassify_merged=True)
        
    Returns:
        List of deduplicated STR predictions
    """
    if not predicted_strs:
        return []
    
    # Filter to only predicted STRs
    str_predictions = [s for s in predicted_strs if s.get('predicted_str', False)]
    
    if not str_predictions:
        logger.info("No STR predictions to deduplicate")
        return predicted_strs
    
    logger.info(f"Starting deduplication of {len(str_predictions)} STR predictions")
    
    # Group by chromosome if required
    if require_same_chromosome:
        by_chromosome = defaultdict(list)
        for i, pred in enumerate(str_predictions):
            chrom = pred.get('chromosome', pred.get('chr', 'unknown'))
            by_chromosome[chrom].append((i, pred))
    else:
        by_chromosome = {'all': [(i, pred) for i, pred in enumerate(str_predictions)]}
    
    merged_strs = []
    processed_indices = set()
    
    for chrom, chrom_predictions in by_chromosome.items():
        logger.debug(f"Processing {len(chrom_predictions)} predictions on {chrom}")
        
        # Sort by position
        def get_start_pos(pred_tuple):
            idx, pred = pred_tuple
            if 'position' in pred:
                return pred['position']
            elif 'start' in pred:
                return int(pred['start'])
            else:
                return 0
        
        sorted_predictions = sorted(chrom_predictions, key=get_start_pos)
        
        # Process in order, merging adjacent/overlapping STRs
        i = 0
        while i < len(sorted_predictions):
            idx1, str1 = sorted_predictions[i]
            
            if idx1 in processed_indices:
                i += 1
                continue
            
            # Get position information for str1
            pos1 = get_start_pos((idx1, str1))
            seq1 = str1.get('sequence', '')
            end1 = pos1 + len(seq1)
            
            # Try to merge with subsequent STRs
            merged_group = [str1]
            merged_indices = [idx1]
            j = i + 1
            
            while j < len(sorted_predictions):
                idx2, str2 = sorted_predictions[j]
                
                if idx2 in processed_indices:
                    j += 1
                    continue
                
                pos2 = get_start_pos((idx2, str2))
                seq2 = str2.get('sequence', '')
                end2 = pos2 + len(seq2)
                
                # Check if they should be merged
                overlap = calculate_overlap(pos1, end1, pos2, end2)
                distance = calculate_distance(pos1, end1, pos2, end2)
                
                should_merge = False
                
                if overlap >= min_overlap:
                    # Overlapping - definitely merge
                    should_merge = True
                    logger.debug(f"Merging overlapping STRs: {pos1}-{end1} and {pos2}-{end2} (overlap: {overlap}bp)")
                elif distance <= max_gap:
                    # Close enough - merge
                    should_merge = True
                    logger.debug(f"Merging adjacent STRs: {pos1}-{end1} and {pos2}-{end2} (gap: {distance}bp)")
                
                if should_merge:
                    merged_group.append(str2)
                    merged_indices.append(idx2)
                    processed_indices.add(idx2)
                    
                    # Update bounds for next iteration
                    pos1 = min(pos1, pos2)
                    end1 = max(end1, end2)
                    j += 1
                else:
                    # Too far apart, stop merging
                    break
            
            # Merge all sequences in the group
            if len(merged_group) == 1:
                # No merging needed
                merged_strs.append(str1)
            else:
                # Merge multiple STRs
                logger.info(f"Merging {len(merged_group)} STRs into one")
                
                # Start with first sequence
                merged_seq = merged_group[0].get('sequence', '')
                merged_pos = get_start_pos((merged_indices[0], merged_group[0]))
                
                # Merge with each subsequent sequence
                for k in range(1, len(merged_group)):
                    seq_k = merged_group[k].get('sequence', '')
                    pos_k = get_start_pos((merged_indices[k], merged_group[k]))
                    
                    merged_seq, merged_pos, merged_end = merge_sequences(
                        merged_seq, merged_pos, seq_k, pos_k
                    )
                
                # Create merged STR prediction
                merged_str = {
                    'sequence': merged_seq,
                    'chromosome': str1.get('chromosome', str1.get('chr', 'unknown')),
                    'position': merged_pos,
                    'start': merged_pos,
                    'end': merged_pos + len(merged_seq),
                    'length': len(merged_seq),
                    'merged_from': len(merged_group),
                    'merged_indices': merged_indices,
                    'original_predictions': [s.get('str_probability', 0.0) for s in merged_group]
                }
                
                # Copy other metadata from first STR
                for key in ['read_name', 'mapping_quality', 'cigar_string', 
                           'is_paired', 'is_proper_pair', 'is_reverse']:
                    if key in str1:
                        merged_str[key] = str1[key]
                
                # Reclassify if requested
                if reclassify_merged and classifier is not None:
                    try:
                        prob = classifier.predict_proba([merged_str])[0]
                        merged_str['str_probability'] = float(prob)
                        merged_str['predicted_str'] = bool(prob >= classifier.threshold)
                        logger.debug(f"Reclassified merged STR: prob={prob:.4f}, predicted={merged_str['predicted_str']}")
                    except Exception as e:
                        logger.warning(f"Failed to reclassify merged STR: {e}")
                        # Use average probability from original predictions
                        merged_str['str_probability'] = sum(merged_str['original_predictions']) / len(merged_str['original_predictions'])
                        merged_str['predicted_str'] = True  # Keep as STR if all originals were STRs
                else:
                    # Use average probability from original predictions
                    merged_str['str_probability'] = sum(merged_str['original_predictions']) / len(merged_str['original_predictions'])
                    merged_str['predicted_str'] = True  # Keep as STR if all originals were STRs
                
                merged_strs.append(merged_str)
            
            processed_indices.add(idx1)
            i += 1
    
    # Add back non-STR predictions
    non_strs = [s for s in predicted_strs if not s.get('predicted_str', False)]
    
    logger.info(f"Deduplication complete: {len(str_predictions)} STRs -> {len(merged_strs)} merged STRs")
    logger.info(f"Total predictions: {len(merged_strs) + len(non_strs)} (was {len(predicted_strs)})")
    
    return merged_strs + non_strs


def deduplicate_str_predictions(
    predictions: List[Dict[str, Any]],
    max_gap: int = 50,
    min_overlap: int = 5,
    reclassify: bool = False,
    classifier: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for detect_and_merge_duplicate_strs.
    
    Args:
        predictions: List of all predictions (STRs and non-STRs)
        max_gap: Maximum gap between STRs to merge (default: 50bp)
        min_overlap: Minimum overlap to consider duplicates (default: 5bp)
        reclassify: Whether to reclassify merged sequences
        classifier: STR_Classifier instance (required if reclassify=True)
        
    Returns:
        Deduplicated list of predictions
    """
    return detect_and_merge_duplicate_strs(
        predictions,
        max_gap=max_gap,
        min_overlap=min_overlap,
        require_same_chromosome=True,
        reclassify_merged=reclassify,
        classifier=classifier
    )

