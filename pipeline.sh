DATA_PATH=/srv/local/data/jp65/CTO/CTTI_new
NCBI_API_KEY='558a8ec64b0df1607941d0261d0a5d273308'
SAVE_PATH=/srv/local/data/jp65/CTO


# Downloading CTTI new data
echo "Downloading CTTI new data"
# python download_ctti.py --save_path $SAVE_PATH 



# # Getting LLM predictions on Pubmed data
echo "Getting LLM predictions on Pubmed data"
cd llm_prediction_on_pubmed 

# echo "Extracting and Updating Pubmed data"
python extract_pubmed_abstracts.py --data_path $DATA_PATH --NCBI_api_key $NCBI_API_KEY --save_path $SAVE_PATH --dev 
# echo "Retrieving top 2 relevant abstracts"
python retrieve_top2_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH --dev
# echo "Obtaining LLM predictions"
python get_llm_predictions.py  --save_path $SAVE_PATH --dev
python clean_and_extract_final_outcomes.py --save_path $SAVE_PATH 