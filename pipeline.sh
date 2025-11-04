DATA_PATH=/srv/local/data/CTO/CTTI_new
SAVE_PATH=/srv/local/data/CTO/nov_2025


# # Downloading CTTI new data
# echo "Downloading CTTI new data"
# python download_ctti.py --save_path $SAVE_PATH 


# ========================= Getting LLM predictions on Pubmed data =========================
# echo "Getting LLM predictions on Pubmed data"
# echo "Extracting and Updating Pubmed data"
# python ./llm_prediction_on_pubmed/extract_pubmed_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# echo "Search Pubmed and extract abstracts"
# python ./llm_prediction_on_pubmed/extract_pubmed_abstracts_through_search.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# echo "Retrieving top 2 relevant abstracts"
# python ./llm_prediction_on_pubmed/retrieve_top2_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# echo "Obtaining LLM predictions"
# python ./llm_prediction_on_pubmed/get_llm_predictions.py  --save_path $SAVE_PATH --azure #--dev
# python ./llm_prediction_on_pubmed/clean_and_extract_final_outcomes.py --save_path $SAVE_PATH


# ========================= Getting Clinical Trial Linkage ========================
# echo "Getting Clinical Trial Linkage"

# echo "Downloading FDA orange book and drug code dictionary"
# python ./clinical_trial_linkage/download_data.py --save_path $SAVE_PATH   # centralize the links in the .sh
# echo "Processing FDA orange book and drug code dictionary"
# python ./clinical_trial_linkage/process_drugbank.py --save_path $SAVE_PATH
# python ./clinical_trial_linkage/create_drug_mapping.py --save_path $SAVE_PATH

# echo "Extracting trial info and trial embeddings"
# python ./clinical_trial_linkage/extract_trial_info.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# python ./clinical_trial_linkage/get_embedding_for_trial_linkage.py --save_path $SAVE_PATH --num_workers 8 --gpu_ids 0,1,2 #--dev


# echo 'Linking Clinical Trials across phases'
# echo 'Phase 4'
# python ./clinical_trial_linkage/create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase4' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 3'
# python ./clinical_trial_linkage/create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2/ Phase 3'
# python ./clinical_trial_linkage/create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2/phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2'
# python ./clinical_trial_linkage/create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2' --num_workers 1 --gpu_ids 4 #--dev

# echo 'Extract outcomes from Clinical Trial Linkage'
# python ./clinical_trial_linkage/extract_outcome_from_trial_linkage.py --save_path $SAVE_PATH
# echo 'Matching with FDA orange book'
# python ./clinical_trial_linkage/match_fda_approvals.py --save_path $SAVE_PATH #--dev


# ========================= News ========================
# skip for now due to quota limits
# python ./news_headlines/get_news.py --mode=get_news --continue_from_prev_log=True --CTTI_PATH=$DATA_PATH --SENTIMENT_MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest" --SAVE_NEWS_LOG_PATH=$SAVE_PATH/news_headlines/ --SAVE_STUDY_NEWS_PATH=$SAVE_PATH/news.csv

# ========================= Stock prices =======================
echo "Updating stock prices and computing slopes"
# Ensure tickers.csv exists under SAVE_PATH (adjust path as needed)
python ./stock_price/get_stocks.py --CTTI_PATH $DATA_PATH --TICKERS_PATH ./stock_price/tickers.csv --SAVE_STOCKS_PATH $SAVE_PATH/stock_data.csv.zip --SAVE_STOCKS_SLOPES_PATH $SAVE_PATH/stock_labels.csv

# ========================= Amendments ========================
python ./stock_price/process_amendments.py --DATA_PATH $DATA_PATH --SAVE_PATH $SAVE_PATH/amendment_counts.csv --years 2

# ========================= Lpdate :abels =================
python labeling/lfs.py --get_thresholds=True --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH --SAVE_PATH=$SAVE_PATH --SKIP_LIST="['new_headlines']"

# # Labeling
# cd ..
# python arrange_labels.py --save_path $SAVE_PATH
