# get_news paths

# HF_HOME="/srv/local/data/chufan2/huggingface/"
CTTI_PATH="/srv/local/data/CTO/CTTI_new/"
# CTTI_PATH="/srv/local/data/CTO/CTTI_old/"

SENTIMENT_MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"
SAVE_NEWS_LOG_PATH="/srv/local/data/CTO/news_headlines/"
SAVE_STUDY_NEWS_PATH="/srv/local/data/CTO/outcome_labels/news.csv"
continue_from_prev_log=True

# python news_headlines/get_news.py --mode=get_news --continue_from_prev_log=$continue_from_prev_log --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH
# python news_headlines/get_news.py --mode=process_news --continue_from_prev_log=$continue_from_prev_log --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH
# python news_headlines/get_news.py --mode=correspond_news_and_studies --continue_from_prev_log=$continue_from_prev_log --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH

LF_EACH_THRESH_PATH="./lf_each_thresh.csv"
HINT_PATH="/srv/local/data/CTO/hint/data/"
LABELS_AND_TICKERS_PATH="/srv/local/data/CTO/outcome_labels/labels_and_tickers.csv"
GPT_PATH="/srv/local/data/CTO/outcome_labels/pubmed_gpt_outcomes.csv"
# GPT_PATH="/home/chufan2/github/CTOD/supplementary/llm_prediction_on_pubmed/pubmed_gpt_outcomes.csv.zip"
LINKAGE_PATH="/srv/local/data/CTO/outcome_labels/Merged_all_trial_linkage_outcome_df__FDA_updated.csv"
# LINKAGE_PATH="/home/chufan2/github/CTOD/supplementary/clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv"
STUDIES_WITH_NEWS_PATH="/srv/local/data/CTO/outcome_labels/news_lfs.csv"
CTO_GOLD_PATH="/srv/local/data/CTO/outcome_labels/final_cto_labels_2020_2024.csv"

# pass in --get_thresholds to get thresholds for first run, then remove it for subsequent runs. It logs the thresholds to a csv file, which is then used in the subsequent runs.
label_mode="RF"
python labeling/lfs.py --get_thresholds=True --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
# python labeling/lfs.py --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
label_mode="LR"
python labeling/lfs.py --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
label_mode="SVM"
python labeling/lfs.py --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
# label_mode="DP"
# python labeling/lfs.py --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
# label_mode="DP_nohint"
# python labeling/lfs.py --LF_EACH_THRESH_PATH=$LF_EACH_THRESH_PATH --CTTI_PATH=$CTTI_PATH --HINT_PATH=$HINT_PATH --LABELS_AND_TICKERS_PATH=$LABELS_AND_TICKERS_PATH --GPT_PATH=$GPT_PATH --LINKAGE_PATH=$LINKAGE_PATH --STUDIES_WITH_NEWS_PATH=$STUDIES_WITH_NEWS_PATH --label_mode=$label_mode --CTO_GOLD_PATH=$CTO_GOLD_PATH
