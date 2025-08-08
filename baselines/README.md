# 🎯 Baseline Models for Clinical Trial Outcome Prediction

## Overview

This module implements various baseline models for clinical trial outcome prediction, providing comprehensive benchmarks against the CTOD weak supervision approach. It includes both classical machine learning models and state-of-the-art transformer-based models.

## 🚀 Quick Start

### Supported Models

#### 🤖 Transformer Models
- **BioBERT**: Biomedical domain pre-trained BERT
- **PubMedBERT**: PubMed and PMC pre-trained BERT
- **ClinicalBERT**: Clinical notes pre-trained BERT

#### 🧮 Classical ML Models
- **Support Vector Machine (SVM)**: Linear and RBF kernels
- **XGBoost**: Gradient boosting trees
- **Random Forest (RF)**: Ensemble decision trees  
- **Logistic Regression (LR)**: Linear classification
- **Multi-Layer Perceptron (MLP)**: Neural network

#### 🔬 Specialized Models
- **SPOT**: Specialized clinical trial outcome model

## 💻 Running the Models

### 1. SPOT Model

```bash
# Update data paths in run_spot.py
python run_spot.py \\
    --train_path <PATH_TO_TRAIN_DATA> \\
    --test_path <PATH_TO_TEST_DATA> \\
    --val_path <PATH_TO_VAL_DATA>
```

### 2. BioBERT Model

```bash
# Configure data paths in biobert_trial_outcome.py
python biobert_trial_outcome.py \\
    --train_path <PATH_TO_TRAIN_DATA> \\
    --test_path <PATH_TO_TEST_DATA> \\
    --val_path <PATH_TO_VAL_DATA> \\
    --model_name "dmis-lab/biobert-base-cased-v1.1"
```

### 3. PubMedBERT Model

```bash
# Configure data paths in pubmedbert_trial_outcome.py  
python pubmedbert_trial_outcome.py \\
    --train_path <PATH_TO_TRAIN_DATA> \\
    --test_path <PATH_TO_TEST_DATA> \\
    --val_path <PATH_TO_VAL_DATA> \\
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
```

### 4. Classical ML Models

```bash
# Run all classical models (SVM, XGBoost, MLP, RF, LR)
python baselines.py \\
    --train_path <PATH_TO_TRAIN_DATA> \\
    --test_path <PATH_TO_TEST_DATA> \\
    --val_path <PATH_TO_VAL_DATA>
```

### 5. Batch Execution

```bash
# Run all models using the provided script
bash run_bert.sh
```

## 📊 Model Configurations

### BERT-based Models

```python
# Default configuration for BERT models
config = {
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'dropout': 0.1
}
```

### Classical ML Models

```python
# Example configurations
models_config = {
    'svm': {
        'C': 1.0,
        'kernel': 'linear',
        'max_iter': 1000
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2
    }
}
```

## 📁 File Structure

```
baselines/
├── README.md                    # This file
├── baselines.py                 # Classical ML models implementation
├── biobert_trial_outcome.py     # BioBERT model training
├── pubmedbert_trial_outcome.py  # PubMedBERT model training  
├── run_spot.py                  # SPOT model implementation
├── run_bert.sh                  # Batch execution script
├── hint_test.ipynb              # Testing and validation notebook
└── results/                     # Model outputs (generated)
    ├── svm_results.json
    ├── biobert_results.json
    └── combined_metrics.csv
```

## 🔧 Data Preparation

### Expected Input Format

All models expect data in the following format:

```python
# Training data structure
train_data = pd.DataFrame({
    'nct_id': ['NCT12345', 'NCT67890'],
    'trial_text': ['Phase 3 study of drug X...', 'Phase 2 trial for...'],
    'label': [1, 0],  # 1: Success, 0: Failure
    'phase': ['Phase 3', 'Phase 2'],
    'sponsor': ['Company A', 'Company B']
})
```

### Feature Engineering

```python
# Text features for classical ML
def extract_text_features(trial_text):
    """Extract features from trial descriptions."""
    features = {
        'length': len(trial_text),
        'num_endpoints': trial_text.count('endpoint'),
        'has_placebo': 'placebo' in trial_text.lower(),
        'num_participants': extract_participant_count(trial_text)
    }
    return features
```

## 📈 Evaluation Metrics

### Primary Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Average Precision**: Area under precision-recall curve
- **Cohen's Kappa**: Inter-rater agreement statistic

### Evaluation by Trial Phase

```python
# Phase-specific evaluation
for phase in ['Phase 1', 'Phase 2', 'Phase 3']:
    phase_results = evaluate_model_by_phase(model, test_data, phase)
    print(f"{phase} - F1: {phase_results['f1']:.3f}, AUC: {phase_results['auc']:.3f}")
```

## 🧪 Experimental Setup

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVM(), param_grid, cv=5, scoring='f1')
```

## 🔗 Integration with CTOD

### Using Weak Supervision Labels

```python
# Combine weak supervision with baselines
from labeling.lfs import generate_weak_labels

# Generate weak labels
weak_labels = generate_weak_labels(trial_data)

# Train baseline with weak labels
model = train_baseline_model(trial_features, weak_labels)
```

### Feature Integration

```python
# Combine multiple feature sources
features = pd.concat([
    text_features,           # Trial text embeddings
    clinical_features,       # Clinical trial metadata
    weak_supervision_features, # From other CTOD modules
    temporal_features        # Time-based features
], axis=1)
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA/GPU Errors with BERT Models**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU if needed
python biobert_trial_outcome.py --device cpu
```

**2. Memory Issues**
```bash
# Reduce batch size
python biobert_trial_outcome.py --batch_size 8

# Use gradient accumulation
python biobert_trial_outcome.py --gradient_accumulation_steps 4
```

**3. Model Loading Errors**
```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Re-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')"
```

### Performance Optimization

```python
# Speed up training
training_args = {
    'fp16': True,                    # Use mixed precision
    'dataloader_num_workers': 4,     # Parallel data loading
    'remove_unused_columns': False,  # Optimize memory
    'gradient_checkpointing': True   # Trade compute for memory
}
```

## 📊 Expected Results

### Baseline Performance (F1-Score)

| Model | Phase 1 | Phase 2 | Phase 3 | Overall |
|-------|---------|---------|---------|---------|
| Random Forest | 0.65 | 0.68 | 0.72 | 0.68 |
| XGBoost | 0.67 | 0.70 | 0.74 | 0.70 |
| BioBERT | 0.71 | 0.73 | 0.77 | 0.74 |
| PubMedBERT | 0.72 | 0.75 | 0.78 | 0.75 |

*Results may vary based on dataset splits and hyperparameters*

### Comparative Analysis

```python
# Generate performance comparison
def compare_baseline_performance(results_dict):
    """Compare performance across all baseline models."""
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df.plot(kind='bar', y=['f1', 'auc', 'accuracy'])
    plt.title('Baseline Model Performance Comparison')
    plt.show()
```

## 📚 Additional Resources

- **BioBERT Paper**: [BioBERT: a pre-trained biomedical language representation model](https://arxiv.org/abs/1901.08746)
- **PubMedBERT Paper**: [Domain-Specific Language Model Pretraining](https://arxiv.org/abs/2007.15779)
- **SPOT Paper**: [Specialized model for clinical trials](https://arxiv.org/abs/2101.10403)
- **Transformers Library**: [Hugging Face Documentation](https://huggingface.co/docs/transformers/)
