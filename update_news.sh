#!bin/bash
# get_news paths

# HF_HOME="/srv/local/data/chufan2/huggingface/"
CTTI_PATH="./CTTI_20241017/"
SENTIMENT_MODEL="yiyanghkust/finbert-tone"
SENTENCE_ENCODER="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SENTENCE_CROSSENCODER="cross-encoder/ms-marco-MiniLM-L-12-v2"
SAVE_NEWS_LOG_PATH="./news_headlines/nct_news_logs"
SAVE_NEWS_EMBEDDING_PATH="./news_headlines/news_title_embeddings.npy"
SAVE_STUDY_TITLE_EMBEDDING_PATH="./news_headlines/studies_title_embeddings.npy"
SAVE_NEWS_PATH="./news_headlines/news.csv"
SAVE_STUDY_NEWS_PATH="./news_headlines/studies_with_news.csv"
NCT_IDS_TO_PROCESS="./manual_labels/nctids.csv"

# SAVE_NEWS_LOG_PATH="./supplementary/news_headlines/news_logs"
# SAVE_NEWS_EMBEDDING_PATH="./supplementary/news_headlines/news_title_embeddings.npy"
# SAVE_STUDY_TITLE_EMBEDDING_PATH="./supplementary/news_headlines/studies_title_embeddings.npy"
# SAVE_NEWS_PATH="./supplementary/news_headlines/news.csv"
# SAVE_STUDY_NEWS_PATH="./supplementary/news_headlines/studies_with_news.csv"

python news_headlines/get_news2.py --mode=get_news --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH --NCT_IDS_TO_PROCESS=$NCT_IDS_TO_PROCESS
# python news_headlines/get_news.py --mode=process_news --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH
# python news_headlines/get_news.py --mode=correspond_news_and_studies --CTTI_PATH=$CTTI_PATH --SENTIMENT_MODEL=$SENTIMENT_MODEL --SENTENCE_ENCODER=$SENTENCE_ENCODER --SAVE_NEWS_LOG_PATH=$SAVE_NEWS_LOG_PATH --SAVE_NEWS_EMBEDDING_PATH=$SAVE_NEWS_EMBEDDING_PATH --SAVE_NEWS_PATH=$SAVE_NEWS_PATH --SAVE_STUDY_NEWS_PATH=$SAVE_STUDY_NEWS_PATH