# 🏷️ Labeling System: Weak Supervision for Clinical Trial Outcomes

## Overview

This module combines all weak supervision sources into final labels using advanced machine learning techniques. It implements our label generation process that merges both unsupervised and supervised approaches to create high-quality training labels for clinical trial outcome prediction.

## 🔧 Prerequisites

### Required Data Sources

1. **CTTI Clinical Trial Data**: Download from [CTTI website](https://aact.ctti-clinicaltrials.org/download)
2. **TOP Dataset**: Clone the publicly available [TOP dataset](https://github.com/futianfan/clinical-trial-outcome-prediction)
   ```bash
   git clone https://github.com/futianfan/clinical-trial-outcome-prediction
   ```
3. **Processed Module Outputs**: Results from other CTOD modules:
   - LLM predictions from `llm_prediction_on_pubmed/`
   - Trial linkages from `clinical_trial_linkage/`
   - News sentiment from `news_headlines/`
   - Stock signals from `stock_price/`

### Dependencies

```bash
pip install snorkel pandas scikit-learn numpy tqdm xgboost
```

## 🚀 Quick Start

### Step 1: Prepare Input Data

Ensure the following paths are correctly set in your scripts:
- Path to CTTI pipe-delimited files
- Path to TOP dataset directory
- Paths to outputs from other CTOD modules

### Step 2: Generate Labels

```bash
cd labeling

# Run the complete label creation workflow
jupyter notebook create_labels.ipynb
```

**Alternative**: Run individual labeling functions:
```bash
python lfs.py --data_path <CTTI_PATH> --save_path <OUTPUT_PATH>
```

## 📊 Labeling Function Architecture

### Core Labeling Functions (LFs)

#### 1. **Clinical Trial Metrics LFs**
```python
# Built-in trial characteristics
lf_results_reported()     # Were results reported?
lf_adverse_events()       # Number of adverse events
lf_trial_amendments()     # Protocol amendments count
lf_enrollment_status()    # Enrollment completion rate
lf_study_duration()       # Trial duration analysis
```

#### 2. **External Source LFs**
```python
# From other CTOD modules
lf_pubmed_gpt()          # GPT predictions on PubMed
lf_trial_linkage()       # Cross-phase linkage signals
lf_news_sentiment()      # News headline sentiment
lf_stock_trends()        # Stock price movement signals
```

### Label Aggregation Strategy

#### Method 1: Data Programming (Snorkel)
```python
from snorkel.labeling.model import LabelModel

# Train label model on LF outputs
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=100)

# Generate training labels
labels_train = label_model.predict(L_train)
```

#### Method 2: Random Forest Meta-Learner
```python
from sklearn.ensemble import RandomForestClassifier

# Train RF on LF outputs as features
rf_labeler = RandomForestClassifier(n_estimators=100, random_state=42)
rf_labeler.fit(lf_outputs_train, ground_truth_labels)

# Generate labels for unlabeled data
predicted_labels = rf_labeler.predict(lf_outputs_unlabeled)
```

## 🎯 Phase-Specific Optimization

### Quantile Threshold Tuning

We determine optimal quantile thresholds for each trial phase:

```python
# Phase-specific threshold optimization
phase_thresholds = {
    'Phase 1': {
        'pubmed_gpt': 0.7,
        'news_sentiment': 0.6,
        'stock_trend': 0.8
    },
    'Phase 2': {
        'pubmed_gpt': 0.75,
        'news_sentiment': 0.65,
        'stock_trend': 0.75
    },
    'Phase 3': {
        'pubmed_gpt': 0.8,
        'news_sentiment': 0.7,
        'stock_trend': 0.7
    }
}
```

### Threshold Selection Process

1. **Grid Search**: Test multiple threshold combinations
2. **Cross-Validation**: Validate on TOP training dataset
3. **Phase Optimization**: Separate optimization for each phase
4. **Performance Metrics**: Optimize for F1-score and coverage

## 📁 Workflow Details

### Complete Label Generation Pipeline

```python
def generate_ctod_labels(data_path, save_path):
    """Complete CTOD label generation pipeline."""
    
    # Step 1: Load all weak supervision sources
    gpt_labels = load_gpt_predictions(f"{save_path}/llm_predictions/")
    linkage_labels = load_trial_linkages(f"{save_path}/clinical_trial_linkage/")
    news_labels = load_news_sentiment(f"{save_path}/news_headlines/")
    stock_labels = load_stock_signals(f"{save_path}/stock_price/")
    
    # Step 2: Apply labeling functions
    lf_matrix = apply_all_lfs(trial_data, external_sources)
    
    # Step 3: Train label model
    label_model = train_snorkel_model(lf_matrix)
    
    # Step 4: Generate final labels
    final_labels = label_model.predict(lf_matrix)
    
    return final_labels
```

### Output Files

```
labeling/output/
├── lf_outputs/                  # Individual LF results
│   ├── lf_results_reported.csv
│   ├── lf_pubmed_gpt.csv
│   └── lf_trial_linkage.csv
├── aggregated_labels/           # Combined label results
│   ├── snorkel_labels.csv       # Data programming results
│   ├── rf_labels.csv            # Random forest results
│   └── final_labels.csv         # Best performing labels
├── evaluation/                  # Label quality metrics
│   ├── lf_analysis.csv          # LF performance analysis
│   └── label_model_metrics.json # Aggregation model performance
└── splits/                      # Pre-split datasets
    ├── pre_2020_labels.csv      # Pre-2020 trials
    └── post_2020_labels.csv     # Post-2020 trials
```

## 🧪 Label Quality Analysis

### LF Performance Metrics

```python
from snorkel.labeling import LFAnalysis

# Analyze individual LF performance
lf_analysis = LFAnalysis(L=lf_matrix, lfs=labeling_functions)
lf_summary = lf_analysis.lf_summary()

print(lf_summary[['Polarity', 'Coverage', 'Overlaps', 'Conflicts']])
```

### Coverage and Conflict Analysis

```python
# Coverage: Fraction of data points labeled by each LF
coverage = (lf_matrix != -1).mean(axis=0)

# Conflicts: Disagreements between LFs
conflicts = snorkel.analysis.get_label_buckets(lf_matrix)
```

## ⚙️ Configuration Options

### Labeling Function Parameters

```python
# Customizable LF parameters
LF_CONFIG = {
    'adverse_events_threshold': 5,      # Min adverse events for negative label
    'amendment_threshold': 3,           # Max amendments for positive label
    'enrollment_completion_rate': 0.8,  # Min completion rate
    'duration_threshold_days': 365,     # Max duration for efficiency
    'confidence_threshold': 0.7         # Min confidence for external LFs
}
```

### Model Training Parameters

```python
# Snorkel Label Model
LABEL_MODEL_CONFIG = {
    'cardinality': 2,           # Binary classification
    'n_epochs': 500,            # Training epochs
    'lr': 0.01,                 # Learning rate
    'l2': 0.0,                  # L2 regularization
    'seed': 42                  # Random seed
}

# Random Forest Meta-Learner
RF_CONFIG = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 10,            # Maximum tree depth
    'min_samples_split': 5,     # Min samples for split
    'random_state': 42          # Random seed
}
```

## 📈 Performance Evaluation

### Label Quality Metrics

```python
def evaluate_label_quality(generated_labels, ground_truth):
    """Evaluate quality of generated labels."""
    metrics = {
        'accuracy': accuracy_score(ground_truth, generated_labels),
        'f1': f1_score(ground_truth, generated_labels),
        'precision': precision_score(ground_truth, generated_labels),
        'recall': recall_score(ground_truth, generated_labels),
        'coverage': len(generated_labels) / len(ground_truth)
    }
    return metrics
```

### Expected Performance

| Labeling Method | Accuracy | F1-Score | Coverage | Precision |
|-----------------|----------|----------|----------|-----------|
| Majority Vote | 0.62 | 0.58 | 0.85 | 0.61 |
| Snorkel LM | 0.68 | 0.64 | 0.88 | 0.67 |
| Random Forest | 0.71 | 0.69 | 0.90 | 0.72 |

## 🔗 Integration Examples

### Using Generated Labels for Training

```python
# Load generated labels
ctod_labels = pd.read_csv('final_labels.csv')

# Train downstream model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(trial_features, ctod_labels['predicted_outcome'])
```

### Combining with Human Labels

```python
# Hybrid approach: CTOD + human labels
def create_hybrid_labels(ctod_labels, human_labels, confidence_threshold=0.8):
    """Combine CTOD weak labels with available human labels."""
    
    # Use human labels where available
    final_labels = human_labels.copy()
    
    # Fill gaps with high-confidence CTOD labels
    high_conf_mask = ctod_labels['confidence'] >= confidence_threshold
    missing_mask = final_labels.isnull()
    
    final_labels[missing_mask & high_conf_mask] = ctod_labels['label']
    
    return final_labels
```

## 🧪 Testing & Validation

### Running Tests

```bash
# Test labeling functions
jupyter notebook hint_test.ipynb

# Validate LF outputs
python -c "
from lfs import test_all_lfs
test_results = test_all_lfs('sample_data.csv')
print('All tests passed:', all(test_results.values()))
"
```

### Custom LF Development

```python
def custom_labeling_function(trial_data):
    """Template for creating custom labeling functions."""
    
    # Your logic here
    labels = []
    for _, trial in trial_data.iterrows():
        if your_condition(trial):
            labels.append(1)  # Positive outcome
        elif your_other_condition(trial):
            labels.append(0)  # Negative outcome  
        else:
            labels.append(-1) # Abstain
    
    return labels
```

## 🐛 Troubleshooting

### Common Issues

**1. Missing Input Data**
```bash
# Verify all module outputs are available
ls -la ../*/output/ | grep csv
```

**2. LF Conflicts**
```python
# Analyze LF agreement
conflict_matrix = snorkel.analysis.get_label_buckets(L_train)
print("High conflict LFs:", find_high_conflict_lfs(conflict_matrix))
```

**3. Low Coverage**
```python
# Check LF coverage
coverage_stats = (L_train != -1).mean(axis=0)
print("LFs with low coverage:", coverage_stats[coverage_stats < 0.1])
```

## 📚 Additional Resources

- **Snorkel Documentation**: [Weak Supervision Framework](https://snorkel.readthedocs.io/)
- **Data Programming**: [Ratner et al., 2016](https://arxiv.org/abs/1605.07723)
- **Weak Supervision Survey**: [Zhou, 2018](https://arxiv.org/abs/1804.09170)