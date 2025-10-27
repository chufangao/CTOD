# CTOD API Documentation

## Core Functions and Classes

### Clinical Trial Linkage Module

#### `support_functions.py`

##### `drug_biologics_nct_ids(intervention_path)`

Extracts NCT IDs for trials involving drug or biological interventions.

**Parameters:**
- `intervention_path` (str): Path to CTTI interventions.txt file

**Returns:**
- `list`: List of NCT IDs for drug and biological intervention trials

**Example:**
```python
from clinical_trial_linkage.support_functions import drug_biologics_nct_ids

nct_ids = drug_biologics_nct_ids('/path/to/ctti/interventions.txt')
print(f"Found {len(nct_ids)} drug/biological trials")
```

##### `extract_study_basic_info(data_path, info_to_extract)`

Extract basic information from the study info file.

**Parameters:**
- `data_path` (str): Path to the study info file
- `info_to_extract` (list): List of information fields to extract
  - Default: `['official_title','start_date','completion_date']`

**Returns:**
- `dict`: Nested dictionary with nct_id as key, extracted information as sub-dictionary

**Example:**
```python
study_info = extract_study_basic_info(
    '/path/to/ctti/studies.txt',
    ['official_title', 'phase', 'enrollment']
)
```

### Labeling Module

#### `lfs.py` - Labeling Functions

##### `reorder_columns(df, cols_in_front)`

Reorder DataFrame columns to place specified columns at the front.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `cols_in_front` (list): Column names to place at the beginning

**Returns:**
- `pd.DataFrame`: DataFrame with reordered columns

##### `lf_results_reported(path)`

Generate labels based on whether trial results were reported.

**Parameters:**
- `path` (str): Path to CTTI data directory

**Returns:**
- `pd.DataFrame`: DataFrame with nct_id and binary label

**Label Logic:**
- `1`: Results were reported (positive signal)
- `0`: Results were not reported (negative signal)

##### `lf_num_sponsors(path, quantile=0.5)`

Generate labels based on number of trial sponsors.

**Parameters:**
- `path` (str): Path to CTTI data directory
- `quantile` (float): Threshold quantile for positive labels

**Returns:**
- `pd.DataFrame`: DataFrame with nct_id and binary label

**Label Logic:**
- `1`: Number of sponsors > quantile threshold
- `0`: Number of sponsors ≤ quantile threshold
- `-1`: Abstain (missing data)

##### `lf_num_patients(path, quantile=0.5)`

Generate labels based on patient enrollment count.

**Parameters:**
- `path` (str): Path to CTTI data directory
- `quantile` (float): Threshold quantile for positive labels

**Returns:**
- `pd.DataFrame`: DataFrame with nct_id and binary label

**Label Logic:**
- Hypothesis: Larger trials (more patients) have higher success probability
- `1`: Patient count > quantile threshold
- `0`: Patient count ≤ quantile threshold

### Data Processing Utilities

#### `download_ctti.py`

Automated CTTI data download using Selenium WebDriver.

**Functionality:**
- Navigates to CTTI pipe files page
- Selects latest available dataset
- Downloads as ZIP file to `./downloads/`

**Requirements:**
- Chrome browser installed
- Selenium WebDriver
- Internet connection

**Usage:**
```bash
python download_ctti.py
```

#### `arrange_labels.py`

Consolidates generated labels from all CTOD modules.

**Command Line Arguments:**
- `--save_path`: Base directory containing module outputs

**Functionality:**
- Copies GPT predictions from LLM module
- Copies trial linkage outcomes
- Organizes all labels in `outcome_labels/` directory

**Usage:**
```bash
python arrange_labels.py --save_path /path/to/results
```

## Data Formats

### Standard Label Format

All labeling functions return DataFrames with consistent structure:

```python
label_df = pd.DataFrame({
    'nct_id': str,      # Clinical trial identifier (required)
    'lf': int,          # Label: 1 (positive), 0 (negative), -1 (abstain)
    'confidence': float, # Optional: prediction confidence [0-1]
    'source': str       # Optional: labeling function source identifier
})
```

### Trial Information Format

```python
trial_info = {
    'NCT12345': {
        'official_title': str,
        'start_date': str,         # Format: YYYY-MM-DD
        'completion_date': str,    # Format: YYYY-MM-DD
        'phase': str,             # e.g., 'Phase 3'
        'enrollment': int,
        'sponsor': str
    }
}
```

### Embedding Format

```python
# Trial embeddings for linkage
embeddings = {
    'nct_id': ['NCT12345', 'NCT67890'],
    'embedding': np.array,      # Shape: (n_trials, embedding_dim)
    'text_source': str,         # Source text for embedding
    'model_used': str          # Embedding model name
}
```

## Error Handling

### Common Exception Types

```python
# File not found errors
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"CTTI data file not found at {data_path}")
    print("Please download CTTI data first using download_ctti.py")

# API errors
try:
    response = openai.Completion.create(...)
except openai.error.RateLimitError:
    print("OpenAI API rate limit exceeded. Please wait and retry.")
except openai.error.AuthenticationError:
    print("Invalid OpenAI API key. Please check your credentials.")

# Data validation errors
if df.empty:
    raise ValueError("No valid trial data found after filtering")
```

### Logging Best Practices

```python
import logging

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)

def process_trials_with_logging(trial_data):
    """Example of proper logging in CTOD functions."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing {len(trial_data)} trials")
    
    try:
        results = expensive_operation(trial_data)
        logger.info(f"Successfully processed {len(results)} trials")
        return results
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
```

## Performance Considerations

### Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(data_path, chunk_size=1000):
    """Process large CTTI datasets efficiently."""
    
    for chunk in pd.read_csv(data_path, sep='|', chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_trials(chunk)
        yield processed_chunk
```

### Parallel Processing

```python
from multiprocessing import Pool
import functools

def parallel_labeling_function(trial_chunk, lf_params):
    """Example of parallelizable labeling function."""
    return [apply_lf(trial, lf_params) for trial in trial_chunk]

# Use with multiprocessing
def run_lf_parallel(trials, lf_params, num_workers=4):
    """Run labeling function in parallel."""
    chunks = np.array_split(trials, num_workers)
    
    with Pool(num_workers) as pool:
        results = pool.map(
            functools.partial(parallel_labeling_function, lf_params=lf_params),
            chunks
        )
    
    return np.concatenate(results)
```

## Testing Utilities

### Validation Functions

```python
def validate_labels(label_df):
    """Validate label DataFrame format and content."""
    
    # Check required columns
    required_cols = ['nct_id', 'lf']
    missing_cols = set(required_cols) - set(label_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check label values
    valid_labels = {-1, 0, 1}
    invalid_labels = set(label_df['lf'].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid label values: {invalid_labels}")
    
    # Check for duplicates
    if label_df['nct_id'].duplicated().any():
        raise ValueError("Duplicate NCT IDs found in labels")
    
    return True

def test_labeling_function(lf_func, test_data_path):
    """Test a labeling function with sample data."""
    
    try:
        result = lf_func(test_data_path)
        validate_labels(result)
        print(f"✓ {lf_func.__name__} passed validation")
        print(f"  Generated {len(result)} labels")
        print(f"  Coverage: {(result['lf'] != -1).mean():.2%}")
        return True
    except Exception as e:
        print(f"✗ {lf_func.__name__} failed: {str(e)}")
        return False
```