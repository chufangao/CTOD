DATA_PATH=/srv/local/data/jp65/CTO/raw_data/76a6hbbrw9v9fpi6oycfq7x6ofgx 
NCBI_API_KEY=''
SAVE_PATH=/srv/local/data/jp65/CTO


# Getting LLM predictions on Pubmed data
echo "Getting LLM predictions on Pubmed data"
cd llm_prediction_on_pubmed 

echo " Extracting and Updating Pubmed data"
python extract_pubmed_abstracts.py --data_path $DATA_PATH --NCBI_api_key $NCBI_API_KEY --save_path $SAVE_PATH --dev 