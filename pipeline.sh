DATA_PATH=/srv/local/data/jp65/CTO/raw_data/76a6hbbrw9v9fpi6oycfq7x6ofgx 
NCBI_API_KEY='558a8ec64b0df1607941d0261d0a5d273308'
SAVE_PATH=/srv/local/data/jp65/CTO


# Downloading CTTI new data
echo "Downloading CTTI new data"
python download_ctti.py --save_path $SAVE_PATH 



# # Getting LLM predictions on Pubmed data
# echo "Getting LLM predictions on Pubmed data"
# cd llm_prediction_on_pubmed 

# echo "Extracting and Updating Pubmed data"
# python extract_pubmed_abstracts.py --data_path $DATA_PATH --NCBI_api_key $NCBI_API_KEY --save_path $SAVE_PATH --dev 
# echo 
# python retrieve_top2_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH --dev
