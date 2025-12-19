# Mutect: Predicting and Visualizing Short Tandem Repeats in Long Read Sequences

A comprehensive pipeline for fetching, processing, and analyzing genomic variants from the NCBI dbVar database, with a focus on short tandem repeats (STRs) and deletion pathogenicity prediction.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quickstart)
5. [Project Structure](#structure)
6. [Usage Examples](#usage)
7. [Data Pipeline](#pipeline)
8. [BAM Processing](#bam)
9. [Visualization](#visualization)

<a name="overview"></a>
## Overview

This project provides tools for:
- **Fetching genomic variants** from NCBI dbVar database with optimized batch API calls
- **Processing and filtering** short tandem repeat (STR) variants
- **Extracting and analyzing** BAM file regions from public genomic datasets
- **Predicting pathogenicity** of genetic deletions using machine learning
- **Visualizing** genomic regions using reference genomes and alignment data

The pipeline is designed for researchers analyzing structural variants, particularly deletions and short tandem repeats, in the context of genetic disease.

<a name="features"></a>
## Features

### Data Acquisition
- **Batch API fetching** from NCBI dbVar (200 variants per request)
- **Intelligent caching** to avoid redundant API calls
- **Rate limiting** to respect NCBI API guidelines
- **Retry logic** for network failures

### Data Processing
- **Variant normalization** across different genome assemblies (GRCh37, GRCh38)
- **Length-based filtering** with customizable thresholds
- **Gene annotation** extraction from dbVar records
- **Clinical significance** parsing and summarization

### BAM File Processing
- **Stream BAM files** directly from URLs (no local download required)
- **Extract specific genomic regions** with coordinate-based queries
- **Parse read metadata** into structured Python objects
- **Export to JSON** or SAM format
- **Calculate read statistics** (mapping quality, paired reads, etc.)

### Machine Learning
- Balanced dataset construction  
- Token-level sequence model  
- Probability-scored predictions  
- CSV/JSON summaries

### Visualization
- Flask + React + TypeScript  
- Displays:
  - Predicted STR loci  
  - REViewer SVGs per locus  
  - ML classifier probabilities  
  - BAM read counts  
  - Locus metadata + motif structure

<a name="installation"></a>
## Installation

### Prerequisites
- Python 3.8+
- Internet connection for API access

### Setup

./setup_and_run.sh
This script will automatically:
1. Check your operating system (macOS, Linux, etc.)
2. Verify required tools are installed (`uv`, `samtools`, `wget`/`curl`, `flask` etc.) and install if not already
3. Download the reference genome (`hs37d5.fa.gz`) if not already present
4. Extract and index the reference genome with `samtools`
5. Run the main pipeline with `uv run main.py`

3. **Configure API credentials**

Create a `.env` file in the project root:
```bash
ENTREZ_EMAIL=your.email@example.com
ENTREZ_API_KEY=your_ncbi_api_key_here  # Optional but recommended
```

Get your NCBI API key from: https://www.ncbi.nlm.nih.gov/account/settings/

<a name="quickstart"></a>
## Quick Start

### Fetch and Process Variants

```python
from src.data.api import DbVarClient
from src.data.data_processor import pass_through_variants
from src.data.preprocessing import filter_str_by_min_length

# Initialize client
client = DbVarClient()

# Fetch variants (uses cache if available)
raw_variants = client.fetch_str_variants(max_results=500)

# Process into standardized format
processed = pass_through_variants(raw_variants)

# Filter by length
filtered = filter_str_by_min_length(processed, min_bp=50)

print(f"Fetched {len(raw_variants)} variants")
print(f"Kept {len(filtered)} after filtering")
```

### Extract BAM Region

```python
from src.input.bam_process import extract_bam_region

# Extract reads from a genomic region
metadata = extract_bam_region(
    bam_file="path/to/file.bam",
    chromosome="chr1",
    start=1000000,
    end=1001000,
    num_lines=20
)

# Access read statistics
stats = metadata.get_read_statistics()
print(f"Total reads: {stats['total_reads']}")
print(f"Avg mapping quality: {stats['avg_mapping_quality']:.2f}")
```

<a name="structure"></a>
## Project Structure

```
2025_3_project_05/
│
├── main.py                     # Top-level pipeline controller
├── src/
│   ├── data/                   # Data utilities & caching
│   ├── ml/                     # STR classifier implementation
│   ├── loci/                   # Loci generator tools
│   ├── bam/                    # Bamlet extraction
│   ├── reviewer/               # REViewer invocation wrapper
│   └── utils/                  # Shared helpers
│
├── scripts/
│   ├── generate_ml_loci.py
│   ├── bamlet_maker.py
│   └── run_reviewer.py
│
├── web/
│   ├── src/backend.py          # Flask backend
│   ├── frontend/               # React + Vite frontend
│   └── README.md
│
├── output/                     # ML outputs / loci / figures
└── data/                       # dbVar + GIAB datasets
```

<a name="usage"></a>
## Usage Examples

### Advanced Variant Fetching

```python
from src.data.api import DbVarClient

client = DbVarClient()

# Fetch with custom size range
variants = client.fetch_str_variants(
    max_results=1000,
    min_size=100,
    max_size=50000
)

# Get statistics
stats = client.get_variant_stats(variants)
print(f"Variants with gene info: {stats['has_gene']}")
print(f"Variants with genomic placement: {stats['has_placement']}")
```

### Length-Based Filtering

```python
from src.data.preprocessing import filter_str_by_min_length

# Filter keeping only variants >= 100bp
filtered = filter_str_by_min_length(
    variants,
    min_bp=100,
    include_unknown=False  # Exclude variants with unknown length
)
```

### BAM Processing with Custom Region

```python
from src.input.bam_process import extract_bam_region

# Extract from GIAB dataset
metadata = extract_bam_region(
    bam_file="ftp://ftp-trace.ncbi.nlm.nih.gov/.../sample.bam",
    chromosome="chr20",
    start=1000000,
    end=1005000,
    num_lines=50,
    return_metadata=True
)

# Save to JSON
with open('bam_region.json', 'w') as f:
    f.write(metadata.to_json())
```

### Command-Line BAM Extraction

```bash
# Extract region and save to JSON
python src/input/bam_process.py \
  path/to/file.bam \
  -c chr1 \
  -s 1000000 \
  -e 1001000 \
  -n 100 \
  --json-output region_data.json

# Stream from URL without printing
python src/input/bam_process.py \
  ftp://example.com/file.bam \
  --no-print \
  --json-output output.json
```

<a name="pipeline"></a>
## Data Pipeline

The complete pipeline flow:

```
┌─────────────────┐
│  NCBI dbVar API │
└────────┬────────┘
         │ Batch fetch (200 variants/request)
         ▼
┌─────────────────┐
│  Cache Layer    │ (./cache/dbvar/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Variants   │ (JSON response)
└────────┬────────┘
         │ Normalize & process
         ▼
┌─────────────────┐
│ Processed Data  │ (Standardized fields)
└────────┬────────┘
         │ Filter by length/criteria
         ▼
┌─────────────────┐
│ Filtered Set    │
└────────┬────────┘
         │
         ├─→ Machine Learning (pathogenicity prediction)
         ├─→ Visualization (genome browser)
         └─→ Export (JSON/CSV)
```

<a name="bam"></a>
## BAM Processing

### Supported Features

- **Indexed BAM files**: Fast random access to specific regions
- **Non-indexed BAM files**: Full-file streaming (slower)
- **FTP/HTTP URLs**: Direct streaming without local download
- **Local files**: Standard file path access

### Read Metadata Structure

Each read is parsed into a `ReadMetadata` object with:

```python
@dataclass
class ReadMetadata:
    query_name: str              # Read identifier
    flag: int                    # SAM flag
    reference_name: str          # Chromosome
    reference_start: int         # 0-based start position
    reference_end: int           # 0-based end position
    mapping_quality: int         # MAPQ score
    cigar_string: str            # CIGAR operations
    query_sequence: str          # DNA sequence
    query_qualities: List[int]   # Phred quality scores
    tags: Dict[str, Any]         # SAM tags (RG, NM, etc.)
    
    # Boolean flags
    is_paired: bool
    is_unmapped: bool
    is_reverse: bool
    is_duplicate: bool
    # ... and more
```

### Example: Analyzing Read Statistics

```python
metadata = extract_bam_region("file.bam", "chr1", 100000, 110000)
stats = metadata.get_read_statistics()

print(f"Mapped reads: {stats['mapped_reads']}")
print(f"Proper pairs: {stats['proper_pairs']}")
print(f"Duplicates: {stats['duplicates']}")
print(f"Average MAPQ: {stats['avg_mapping_quality']:.2f}")
```

<a name="visualization"></a>
## Visualization

### Genome Browser Integration

```python
from src.visualization.main import GenomeReferenceVisualizer

viz = GenomeReferenceVisualizer(
    reference_fasta="hs37d5.fa",
    data_dir="./genomic_data"
)

# Visualize a gene region
viz.visualize_gene(
    sample_id="HG00096",
    gene_name="BRCA1",
    output_dir="./visualizations"
)

# Custom region
viz.visualize_custom_region(
    sample_id="HG00096",
    chrom="17",
    start=41200000,
    end=41250000
)
```

<a name="development"></a>
## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

We follow PEP 8 guidelines. Format your code with:

```bash
black src/
flake8 src/
```

### Performance Optimization

**API Fetching:**
- Before: 500 variants = 500 requests ≈ 167 seconds
- After: 500 variants = 3 batches ≈ 3 seconds
- With cache: Subsequent runs < 1 second

**BAM Streaming:**
- Indexed BAM: Random access to any region (fast)
- Non-indexed: Full file streaming (slow but works)

### Caching

The API client caches results in `./cache/dbvar/`. To clear cache:

```bash
rm -rf ./cache/dbvar/
```

Or disable caching:

```python
client = DbVarClient(cache_dir=None)
variants = client.fetch_str_variants(use_cache=False)
```

## Acknowledgments

- NCBI dbVar for genomic variant data
- 1000 Genomes Project for reference data
- GIAB Consortium for validation datasets
- Biopython team for Entrez utilities

## Developers

- Brandon Tang 
- Dennis Kritchko
- Dixon Snider
- Satvik Garg
- Winston Thov

Huge thank you to Jan Fridman at UBC for assisting in all the biological components of this project 