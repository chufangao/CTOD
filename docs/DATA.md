# CTOD Data Documentation

## 📊 Dataset Overview

The Clinical Trial Outcome Detection (CTOD) project integrates multiple data sources to create a comprehensive benchmark for predicting clinical trial outcomes. This document provides detailed information about all data sources, formats, and processing pipelines.

## 🗃️ Primary Data Sources

### 1. CTTI (Clinical Trials Transformation Initiative)
**Source**: [https://aact.ctti-clinicaltrials.org/](https://aact.ctti-clinicaltrials.org/)

**Description**: Comprehensive database of clinical trials from ClinicalTrials.gov

**Key Files:**
- `studies.txt` - Basic trial information
- `interventions.txt` - Trial interventions and drugs
- `sponsors.txt` - Trial sponsors and collaborators  
- `calculated_values.txt` - Computed trial metrics
- `outcome_counts.txt` - Outcome measurement counts

**Format**: Pipe-delimited text files (`|` separator)

**Size**: ~500MB compressed, ~2GB extracted

**Update Frequency**: Daily

### 2. TOP Dataset (Baseline Comparison)
**Source**: [https://github.com/futianfan/clinical-trial-outcome-prediction](https://github.com/futianfan/clinical-trial-outcome-prediction)

**Description**: Manually labeled clinical trial outcomes for validation

**Key Features:**
- Expert-curated ground truth labels
- Multi-phase trial coverage
- Used for evaluation and threshold tuning

**Format**: CSV files with trial IDs and binary outcomes

### 3. PubMed/PMC Literature
**Source**: [https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)

**Description**: Biomedical literature abstracts linked to clinical trials

**Access Method**: NCBI E-utilities API

**Data Collected:**
- Abstract text from linked publications
- Publication metadata (authors, journal, date)
- Medical Subject Headings (MeSH) terms

### 4. FDA Orange Book
**Source**: [https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files](https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files)

**Description**: FDA-approved drug products database

**Usage**: Match successful Phase 3 trials to drug approvals

**Format**: CSV files with drug names, approval dates, and company information

### 5. News Headlines
**Source**: Google News (via GNews/SerpAPI)

**Description**: Financial and medical news related to pharmaceutical companies

**Collection Strategy:**
- Target top 1000 industry sponsors
- Time window around trial completion dates
- Filter for relevance using keyword matching

### 6. Stock Price Data
**Source**: Yahoo Finance (via yfinance library)

**Description**: Historical stock prices for publicly traded pharmaceutical companies

**Features Extracted:**
- 5-day Simple Moving Average (SMA)
- 7-day trend slopes post-trial completion
- Trading volume and volatility metrics

## 📋 Data Processing Pipeline

### Stage 1: Data Collection
```
Raw Data Sources → Download Scripts → Local Storage
│
├── CTTI Data (download_ctti.py)
├── PubMed Abstracts (extract_pubmed_abstracts.py)
├── News Headlines (get_news.py)
├── Stock Prices (get_stocks.py)
└── FDA Data (download_data.py)
```

### Stage 2: Feature Extraction
```
Raw Data → Feature Engineering → Embeddings & Signals
│
├── Trial Text → PubMedBERT Embeddings
├── News Headlines → FinBERT Sentiment
├── Stock Prices → Trend Slopes
└── Clinical Metrics → Quantified Features
```

### Stage 3: Weak Supervision
```
Features → Labeling Functions → Weak Labels
│
├── Clinical LFs (trial metrics)
├── Literature LFs (PubMed + LLM)
├── Market LFs (news + stock)
└── Linkage LFs (cross-phase connections)
```

### Stage 4: Label Aggregation
```
Weak Labels → Meta-Learning → Final Labels
│
├── Snorkel Data Programming
├── Random Forest Meta-Learner
└── Threshold Optimization
```

## 📁 Data Directory Structure

### Recommended Organization

```
ctod_data/
├── raw/                           # Original downloaded data
│   ├── ctti/                      # CTTI pipe-delimited files
│   │   ├── studies.txt
│   │   ├── interventions.txt
│   │   ├── sponsors.txt
│   │   └── ...
│   ├── fda/                       # FDA Orange Book data
│   ├── top/                       # TOP dataset for evaluation
│   └── external/                  # Additional external datasets
│
├── processed/                     # Intermediate processed data
│   ├── embeddings/                # Trial and text embeddings
│   ├── features/                  # Extracted features
│   ├── matches/                   # News-trial, linkage matches
│   └── abstracts/                 # PubMed abstracts
│
├── labels/                        # Generated weak labels
│   ├── individual_lfs/            # Individual labeling function outputs
│   ├── aggregated/                # Combined labels
│   └── splits/                    # Train/test/validation splits
│
└── results/                       # Final outputs and models
    ├── models/                    # Trained models
    ├── predictions/               # Model predictions
    └── evaluation/                # Performance metrics
```

## 📊 Data Formats and Schemas

### Trial Information Schema

```python
# Core trial information structure
trial_schema = {
    'nct_id': str,                 # Primary key (e.g., 'NCT01234567')
    'official_title': str,         # Full trial title
    'brief_title': str,            # Short trial title
    'phase': str,                  # Trial phase (e.g., 'Phase 3')
    'start_date': str,             # Format: 'YYYY-MM-DD'
    'completion_date': str,        # Format: 'YYYY-MM-DD'
    'enrollment': int,             # Number of participants
    'sponsor': str,                # Primary sponsor name
    'intervention_type': str,      # 'Drug', 'Biological', etc.
    'study_type': str,             # 'Interventional', 'Observational'
    'allocation': str,             # 'Randomized', 'Non-Randomized'
    'masking': str,                # 'Double Blind', 'Open Label', etc.
    'primary_purpose': str         # 'Treatment', 'Prevention', etc.
}
```

### Label Format

```python
# Standard label format across all modules
label_schema = {
    'nct_id': str,                 # Trial identifier
    'label': int,                  # 1: Success, 0: Failure, -1: Abstain
    'confidence': float,           # [0-1] prediction confidence
    'source': str,                 # Labeling function identifier
    'phase': str,                  # Trial phase
    'created_date': str,           # Label creation timestamp
    'metadata': dict               # Additional source-specific info
}
```

### Embedding Format

```python
# Embeddings for trial similarity and linkage
embedding_schema = {
    'nct_id': str,                 # Trial identifier
    'embedding': List[float],      # Dense vector representation
    'embedding_model': str,        # Model used (e.g., 'PubMedBERT')
    'text_source': str,            # Source text for embedding
    'dimension': int,              # Embedding dimensionality
    'normalization': str           # 'l2', 'none', etc.
}
```

## 🔄 Data Flow Diagrams

### Complete Pipeline Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Feature         │───▶│ Weak Labels     │
│   Sources       │    │  Extraction      │    │ Generation      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│                      │                       │
├─ CTTI Data           ├─ Text Embeddings      ├─ Clinical LFs
├─ PubMed Abstracts    ├─ Sentiment Scores     ├─ Literature LFs  
├─ News Headlines      ├─ Stock Trends         ├─ Market LFs
├─ Stock Prices        ├─ Trial Linkages       └─ Linkage LFs
└─ FDA Approvals       └─ Clinical Metrics     
                                               
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Weak Labels     │───▶│   Label          │───▶│ Final Training  │
│ Aggregation     │    │   Model          │    │ Labels          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│                      │                       │
├─ Snorkel LM          ├─ Random Forest        ├─ Binary Outcomes
├─ Majority Vote       ├─ Threshold Opt       ├─ Confidence Scores
└─ Weighted Avg        └─ Cross Validation     └─ Phase-specific
```

## 📏 Data Quality Metrics

### Coverage Statistics

| Data Source | Coverage | Quality | Latency |
|-------------|----------|---------|---------|
| CTTI Core | 100% | High | Current |
| PubMed Links | ~60% | High | 6-12 months |
| News Headlines | ~30% | Medium | Real-time |
| Stock Prices | ~25% | High | Real-time |
| FDA Approvals | ~15% | Very High | 6-24 months |

### Data Quality Checks

```python
def validate_trial_data(trial_df):
    """Comprehensive data quality validation."""
    
    quality_report = {
        'total_trials': len(trial_df),
        'missing_nct_ids': trial_df['nct_id'].isnull().sum(),
        'missing_titles': trial_df['official_title'].isnull().sum(),
        'missing_phases': trial_df['phase'].isnull().sum(),
        'invalid_dates': validate_date_format(trial_df),
        'duplicate_ids': trial_df['nct_id'].duplicated().sum(),
        'coverage_by_phase': trial_df.groupby('phase').size()
    }
    
    return quality_report
```

## 🔄 Data Updates and Maintenance

### Regular Update Schedule

1. **CTTI Data**: Weekly updates recommended
2. **News Headlines**: Daily for active monitoring
3. **Stock Prices**: Daily during market hours
4. **FDA Approvals**: Monthly or quarterly
5. **PubMed Links**: Bi-weekly for new publications

### Update Scripts

```bash
# Update CTTI data
python download_ctti.py --update

# Update news headlines
bash update_news.sh

# Update labels with new data
bash update_labels.sh
```

## 💾 Data Storage Recommendations

### Local Development
- **SSD Storage**: Recommended for better I/O performance
- **Backup Strategy**: Regular backups of processed data
- **Version Control**: Track data versions and processing parameters

### Production Environment
- **Distributed Storage**: HDFS, AWS S3, or Google Cloud Storage
- **Database Integration**: PostgreSQL or MongoDB for structured queries
- **Caching Strategy**: Redis or Memcached for frequent access patterns

### Data Retention Policy

```python
# Example retention policy
retention_policy = {
    'raw_data': '2 years',           # Keep original downloads
    'processed_features': '1 year',  # Feature engineering outputs
    'intermediate_results': '6 months', # Temporary processing files
    'final_labels': 'permanent',     # Keep all final outputs
    'model_checkpoints': '1 year',   # Trained model files
    'logs': '3 months'               # Processing and error logs
}
```

## 🔍 Data Privacy and Ethics

### Privacy Considerations

- **Clinical Trial Data**: All data is publicly available from ClinicalTrials.gov
- **No PHI**: No Protected Health Information (PHI) is included
- **Aggregated Results**: Individual patient data is not accessible
- **Compliance**: Follows NIH and FDA data sharing guidelines

### Ethical Use Guidelines

1. **Research Purpose**: Use only for legitimate research purposes
2. **Attribution**: Cite original data sources and CTOD paper
3. **No Re-identification**: Do not attempt to identify individuals
4. **Responsible Sharing**: Follow institutional data sharing policies

## 📚 Additional Resources

### Data Documentation
- **CTTI Schema**: [Database Schema Documentation](https://aact.ctti-clinicaltrials.org/schema)
- **ClinicalTrials.gov**: [Data Element Definitions](https://clinicaltrials.gov/ct2/about-studies/glossary)
- **FDA Orange Book**: [User Guide](https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files)

### Related Datasets
- **TOP Dataset**: [Clinical Trial Outcome Prediction](https://github.com/futianfan/clinical-trial-outcome-prediction)
- **DrugBank**: [Drug Information Database](https://go.drugbank.com/)
- **PubChem**: [Chemical Information](https://pubchem.ncbi.nlm.nih.gov/)

### Analysis Tools
- **Pandas**: [Data Analysis Library](https://pandas.pydata.org/)
- **Snorkel**: [Weak Supervision Framework](https://snorkel.org/)
- **Transformers**: [NLP Models](https://huggingface.co/transformers/)