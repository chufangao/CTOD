#!/bin/bash

# CTOD Complete Pipeline Script
# 
# This script runs the complete Clinical Trial Outcome Detection pipeline,
# processing data from multiple sources to generate weak supervision labels
# for clinical trial outcome prediction.
#
# Prerequisites:
# - All required Python packages installed (see requirements.txt)
# - API keys configured (NCBI, OpenAI, SerpAPI)
# - Sufficient disk space (50GB+) and memory (16GB+)
#
# Usage:
#   export DATA_PATH=/path/to/ctti/data
#   export SAVE_PATH=/path/to/save/results
#   bash pipeline.sh
#
# Estimated Runtime: 2-7 days depending on hardware and dataset size

# Configuration - Update these paths for your setup
DATA_PATH=/srv/local/data/CTO/CTTI_new
SAVE_PATH=/srv/local/data/CTO

echo "=== CTOD Pipeline Starting ==="
echo "Data Path: $DATA_PATH"
echo "Save Path: $SAVE_PATH"
echo "Start Time: $(date)"

# Uncomment sections as needed for your workflow

# =============================================================================
# STAGE 1: DATA DOWNLOAD AND PREPARATION
# =============================================================================

# # Download CTTI dataset (uncomment if needed)
# echo "=== Downloading CTTI Data ==="
# python download_ctti.py --save_path $SAVE_PATH 

# =============================================================================  
# STAGE 2: LLM PREDICTIONS ON PUBMED DATA
# =============================================================================

# Getting LLM predictions on Pubmed data
echo "=== Processing LLM Predictions on PubMed Data ==="
cd llm_prediction_on_pubmed 

# # Extract PubMed abstracts (uncomment to run)
# echo "Extracting and Updating Pubmed data"
# python extract_pubmed_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev 

# # Alternative: Search-based abstract extraction
# echo "Search Pubmed and extract abstracts"
# python extract_pubmed_abstracts_through_search.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev

# # Get top 2 most relevant abstracts per trial
# echo "Retrieving top 2 relevant abstracts"
# python retrieve_top2_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev

# # Generate LLM predictions using GPT-3.5
# echo "Obtaining LLM predictions"
# python get_llm_predictions.py  --save_path $SAVE_PATH --azure #--dev

# # Clean and finalize LLM outcomes
# python clean_and_extract_final_outcomes.py --save_path $SAVE_PATH 

# =============================================================================
# STAGE 3: CLINICAL TRIAL LINKAGE GENERATION  
# =============================================================================

# Getting Clinical Trial Linkage
echo "=== Processing Clinical Trial Linkage ==="
cd ..
cd clinical_trial_linkage

# # Download external datasets (FDA Orange Book, DrugBank, etc.)
# echo "Downloading FDA orange book and drug code dictionary"
# python download_data.py --save_path $SAVE_PATH

# # Process drug databases for name standardization
# echo "Processing FDA orange book and drug code dictionary"
# python process_drugbank.py --save_path $SAVE_PATH
# python create_drug_mapping.py --save_path $SAVE_PATH

# # Extract trial information and generate embeddings
# echo "Extracting trial info and trial embeddings"
# python extract_trial_info.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# python get_embedding_for_trial_linkage.py --save_path $SAVE_PATH --num_workers 8 --gpu_ids 0,1,2 #--dev

# # Link trials across different phases (from later to earlier phases)
# echo 'Linking Clinical Trials across phases'
# echo 'Phase 4'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase4' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 3'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2/ Phase 3'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2/phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2' --num_workers 1 --gpu_ids 4 #--dev

# # Extract outcomes from trial linkages
# echo 'Extract outcomes from Clinical Trial Linkage'
# python extract_outcome_from_trial_linkage.py --save_path $SAVE_PATH 

# # Match with FDA approvals for additional validation
# echo 'Matching with FDA orange book'
# python match_fda_approvals.py --save_path $SAVE_PATH #--dev

# =============================================================================
# STAGE 4: NEWS HEADLINES AND STOCK PRICE ANALYSIS (Optional)
# =============================================================================

# Note: These modules require additional setup and long processing times
# See respective module READMEs for detailed instructions

# News Headlines Processing (requires SerpAPI, takes weeks)
# cd ../news_headlines  
# python get_news.py --mode=get_news

# Stock Price Analysis (requires ticker data)
# cd ../stock_price
# jupyter notebook slope_calculation.ipynb

# =============================================================================
# STAGE 5: LABEL AGGREGATION AND FINALIZATION
# =============================================================================

# Copy all labeling results to centralized location
echo "=== Organizing Final Labels ==="
cd ..
python arrange_labels.py --save_path $SAVE_PATH

echo "=== CTOD Pipeline Complete ==="
echo "End Time: $(date)"
echo "Results saved to: $SAVE_PATH/outcome_labels/"

# =============================================================================
# OPTIONAL: RUN BASELINE MODELS
# =============================================================================

# echo "=== Running Baseline Models ==="
# cd baselines
# python baselines.py --train_path $SAVE_PATH/labels/train.csv --test_path $SAVE_PATH/labels/test.csv