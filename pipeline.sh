DATA_PATH=/srv/local/data/CTO/CTTI_new
SAVE_PATH=/srv/local/data/CTO


# # Downloading CTTI new data
# echo "Downloading CTTI new data"
# python download_ctti.py --save_path $SAVE_PATH 



# # # Getting LLM predictions on Pubmed data
echo "Getting LLM predictions on Pubmed data"
cd llm_prediction_on_pubmed 

# echo "Extracting and Updating Pubmed data"
# python extract_pubmed_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev 
# echo "Search Pubmed and extract abstracts"
# python extract_pubmed_abstracts_through_search.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# echo "Retrieving top 2 relevant abstracts"
# python retrieve_top2_abstracts.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# echo "Obtaining LLM predictions"
# python get_llm_predictions.py  --save_path $SAVE_PATH --azure #--dev
# python clean_and_extract_final_outcomes.py --save_path $SAVE_PATH 


# # # # Getting Clinical Trial Linkage
# echo "Getting Clinical Trial Linkage"
cd ..
cd clinical_trial_linkage

# echo "Downloading FDA orange book and drug code dictionary"
# python download_data.py --save_path $SAVE_PATH   # centralize the links in the .sh
# echo "Processing FDA orange book and drug code dictionary"
# python process_drugbank.py --save_path $SAVE_PATH
# python create_drug_mapping.py --save_path $SAVE_PATH

# echo "Extracting trial info and trial embeddings"
# python extract_trial_info.py --data_path $DATA_PATH --save_path $SAVE_PATH #--dev
# python get_embedding_for_trial_linkage.py --save_path $SAVE_PATH --num_workers 8 --gpu_ids 0,1,2 #--dev


# echo 'Linking Clinical Trials across phases'
# echo 'Phase 4'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase4' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 3'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2/ Phase 3'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2/phase3' --num_workers 1 --gpu_ids 4 #--dev
# echo 'Phase 2'
# python create_trial_linkage.py --save_path $SAVE_PATH --target_phase 'phase2' --num_workers 1 --gpu_ids 4 #--dev

# echo 'Extract outcomes from Clinical Trial Linkage'
# python extract_outcome_from_trial_linkage.py --save_path $SAVE_PATH 
# echo 'Matching with FDA orange book'
# python match_fda_approvals.py --save_path $SAVE_PATH #--dev



# News


#Stock prices


# Labeling
# echo "Copy all labeling results to the labeling folder"
cd ..
python arrange_labels.py --save_path $SAVE_PATH

# limit it to drugs