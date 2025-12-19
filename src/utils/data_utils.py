import json
from pathlib import Path
from typing import Any, Dict, List


def load_labeled_data(json_path: str) -> List[Dict[str, Any]]:
    """Load pre-labeled sequence data from JSON file.
    
    Args:
        json_path: Path to JSON file with labeled sequences
        
    Returns:
        List of dictionaries with 'sequence' and 'is_str' keys
    """
    print(f"Loading data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    
    # Handle different JSON formats
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Get sequence from various possible keys
                seq = item.get('sequence') or item.get('query_sequence') or item.get('seq')
                if seq:
                    # Preserve is_str label if it exists, otherwise default to False
                    is_str = item.get('is_str', False)
                    sequences.append({
                        'sequence': seq,
                        'is_str': is_str,
                        # Preserve other metadata
                        **{k: v for k, v in item.items() if k not in ['sequence', 'query_sequence', 'seq', 'is_str']}
                    })
            elif isinstance(item, str):
                # Just a sequence string
                sequences.append({'sequence': item, 'is_str': False})
    
    elif isinstance(data, dict):
        if 'reads' in data:
            # BamRegionMetadata format
            for read in data['reads']:
                seq = read.get('query_sequence')
                if seq:
                    is_str = read.get('is_str', False)
                    sequences.append({
                        'sequence': seq,
                        'is_str': is_str,
                        **{k: v for k, v in read.items() if k not in ['query_sequence', 'is_str']}
                    })
        elif 'sequence' in data or 'query_sequence' in data:
            # Single sequence object
            seq = data.get('sequence') or data.get('query_sequence')
            if seq:
                is_str = data.get('is_str', False)
                sequences.append({
                    'sequence': seq,
                    'is_str': is_str,
                    **{k: v for k, v in data.items() if k not in ['sequence', 'query_sequence', 'is_str']}
                })
    
    if not sequences:
        raise ValueError(f"No valid sequences found in {json_path}")
    
    print(f"  Loaded {len(sequences)} sequences")
    labeled_str = sum(1 for s in sequences if s.get('is_str') == True)
    labeled_non_str = sum(1 for s in sequences if s.get('is_str') == False)
    print(f"  Labels: {labeled_str} STR, {labeled_non_str} non-STR")
    
    return sequences

def create_balanced_dataset(str_sequences: List[Dict], normal_sequences: List[Dict], 
                           output_path: str = "output/balanced_dataset.json") -> List[Dict]:
    """Create a balanced dataset with equal STR and non-STR sequences.
    
    Args:
        str_sequences: List of sequences labeled as STRs
        normal_sequences: List of sequences labeled as non-STRs
        output_path: Where to save the balanced dataset
        
    Returns:
        Balanced list of sequences
    """
    print(f"\n{'='*80}")
    print(f"CREATING BALANCED DATASET")
    print(f"{'='*80}")
    print(f"STR sequences: {len(str_sequences)}")
    print(f"Normal sequences: {len(normal_sequences)}")
    
    # Make copies to avoid modifying originals
    import copy
    str_seqs_copy = copy.deepcopy(str_sequences)
    normal_seqs_copy = copy.deepcopy(normal_sequences)
    
    # Label the sequences
    for seq in str_seqs_copy:
        seq['is_str'] = True
    for seq in normal_seqs_copy:
        seq['is_str'] = False
    
    # Balance the dataset
    min_count = min(len(str_seqs_copy), len(normal_seqs_copy))
    
    if min_count == 0:
        print("WARNING: Cannot create balanced dataset - one class has 0 sequences")
        return str_seqs_copy + normal_seqs_copy
    
    balanced = str_seqs_copy[:min_count] + normal_seqs_copy[:min_count]
    
    print(f"Balanced dataset size: {len(balanced)} ({min_count} STRs, {min_count} non-STRs)")
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(balanced, f, indent=2, default=str)  # Added default=str for non-serializable types
    print(f"Saved to: {output_path}")
    
    return balanced

def load_sequences_for_prediction(json_path: str) -> List[Dict[str, Any]]:
    """Load sequences for prediction from JSON file (no labels required).
    
    Args:
        json_path: Path to JSON file with sequences
        
    Returns:
        List of dictionaries with 'sequence' key
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    
    # Handle different JSON formats
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                seq = item.get('sequence') or item.get('query_sequence')
                if seq:
                    sequences.append({'sequence': seq, **item})
            elif isinstance(item, str):
                sequences.append({'sequence': item})
    elif isinstance(data, dict) and 'reads' in data:
        # BamRegionMetadata format
        for read in data['reads']:
            seq = read.get('query_sequence')
            if seq:
                sequences.append({'sequence': seq, **read})
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")
    
    return sequences