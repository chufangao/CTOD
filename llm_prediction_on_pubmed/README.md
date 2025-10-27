# 🧠 LLM Predictions on PubMed Abstracts

<p align="center">
  <img src="LLM_prediction_method.png" alt="LLM Prediction Method" width="600"/>
</p>

## Overview

This module leverages PubMed abstracts linked to clinical trials to predict trial outcomes using Large Language Models (LLMs). We use GPT-3.5 with carefully designed prompts to analyze medical literature and generate weak supervision labels for clinical trial success prediction.

## 🔧 Prerequisites

### Required Data & API Access

1. **CTTI Clinical Trial Data**: Download from [CTTI website](https://aact.ctti-clinicaltrials.org/download)
2. **NCBI API Key**: Required for PubMed access
   - Create account: [NCBI Account Setup](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us)
   - Obtain API key from your NCBI account settings
3. **OpenAI API Key**: For GPT-3.5 predictions
   - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Ensure sufficient credits for large-scale processing

### Dependencies

```bash
pip install openai pandas numpy tqdm sentence-transformers requests beautifulsoup4
```

## 🚀 Quick Start

### Step 1: Extract PubMed Abstracts

```bash
cd llm_prediction_on_pubmed

# Extract all linked PubMed abstracts
python extract_pubmed_abstracts.py \\
    --data_path <PATH_TO_CTTI_DATA> \\
    --NCBI_api_key <YOUR_NCBI_API_KEY> \\
    --save_path <SAVE_PATH>
```

**Alternative Method** (using search):
```bash
# Extract abstracts through PubMed search
python extract_pubmed_abstracts_through_search.py \\
    --data_path <PATH_TO_CTTI_DATA> \\
    --save_path <SAVE_PATH>
```

### Step 2: Retrieve Top 2 Most Relevant Abstracts

```bash
# Get the 2 most relevant abstracts per trial
python retrieve_top2_abstracts.py \\
    --data_path <PATH_TO_CTTI_DATA> \\
    --save_path <SAVE_PATH>
```

**Output**: `<save_path>/top_2_extracted_pubmed_articles.csv`

### Step 3: Generate LLM Predictions

```bash
# Get GPT-3.5 predictions on abstracts
python get_llm_predictions.py \\
    --top_2_pubmed_path <PATH_TO_TOP2_CSV> \\
    --save_path <SAVE_PATH> \\
    --openai_api_key <YOUR_OPENAI_KEY>

# Alternative: Use Azure OpenAI
python get_llm_predictions.py \\
    --top_2_pubmed_path <PATH_TO_TOP2_CSV> \\
    --save_path <SAVE_PATH> \\
    --azure
```

### Step 4: Process Final Outcomes

```bash
# Combine and clean all predictions
python clean_and_extract_final_outcomes.py \\
    --gpt_decisions_path <PATH_TO_LLM_PREDICTIONS> \\
    --top_2_pubmed_path <PATH_TO_TOP2_CSV>
```

**Final Output**: `<top_2_pubmed_path>/pubmed_gpt_outcomes.csv`

## 📊 Methodology

### Abstract Selection Strategy

1. **Filtering**: Focus on "Derived" and "Results" category abstracts
2. **Ranking**: Use title similarity to official trial title
3. **Selection**: Pick top 2 most relevant abstracts per trial
4. **Validation**: Cross-reference with trial metadata

### LLM Prompting Strategy

Our prompts are designed to:
- Extract key outcome indicators from abstracts
- Identify success/failure signals in medical language
- Handle uncertainty and incomplete information
- Provide reasoning for predictions

## 🔧 Configuration Options

### Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--data_path` | Path to CTTI dataset | - | Yes |
| `--save_path` | Output directory | - | Yes |
| `--NCBI_api_key` | NCBI API key for PubMed | - | Yes* |
| `--openai_api_key` | OpenAI API key | - | Yes* |
| `--azure` | Use Azure OpenAI instead | False | No |
| `--dev` | Use smaller dataset for testing | False | No |

*API keys can also be set as environment variables

### Environment Variables

```bash
export NCBI_API_KEY="your_ncbi_key"
export OPENAI_API_KEY="your_openai_key"
```

## 📈 Performance Metrics

### Processing Statistics

- **Abstracts per Trial**: Average 2-5 relevant abstracts
- **Processing Speed**: ~100 trials/hour (with API rate limits)
- **Success Rate**: 85-90% abstract retrieval success
- **LLM Accuracy**: Reported in main paper

### Resource Requirements

- **Memory**: 4-8GB RAM recommended
- **Storage**: 2-5GB for abstract cache
- **API Costs**: ~$50-100 for full dataset processing
- **Processing Time**: 12-24 hours for complete pipeline

## 🧪 Testing & Development

### Test with Small Dataset

```bash
# Run on subset for testing
python extract_pubmed_abstracts.py --dev --data_path <PATH> --save_path <PATH>
```

### Debugging Common Issues

1. **API Rate Limits**: 
   - NCBI: 3 requests/second (with key), 1/second (without)
   - OpenAI: Varies by plan and model

2. **Abstract Not Found**: 
   - Check PMID validity
   - Verify PubMed availability
   - Consider alternative search strategies

3. **LLM Prediction Errors**:
   - Check API key validity
   - Monitor token usage and costs
   - Review prompt formatting

### Output File Structure

```
save_path/
├── extracted_abstracts/          # Raw PubMed abstracts
├── top_2_extracted_pubmed_articles.csv  # Filtered top abstracts
├── llm_predictions/              # GPT predictions and features
├── qa_pairs/                     # Question-answer pairs
└── pubmed_gpt_outcomes.csv       # Final outcome predictions
```

## 🔗 Integration with Other Modules

This module's outputs are used by:
- **[Labeling Module](../labeling/)**: Combines with other weak supervision sources
- **[Baselines](../baselines/)**: Provides features for baseline models

## 📚 Additional Resources

- **Prompt Engineering**: See paper appendix for detailed prompt designs
- **PubMed API**: [NCBI E-utilities documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- **OpenAI API**: [OpenAI documentation](https://platform.openai.com/docs/)
