'''
Psuedo code for the GPT-4 extraction model
1. Extract pubmed articles
2. Filter out articles
    - Filter articles in the frame of a year after the trial and anytime before the trial
    - Only consider derived and results articles.
3. Extract top 2 similar article based on trial title and abstract title
4. Feed to GPT to obtain following:
    - description of the trial
    - Features of the trial
    - Outcome prediction ('success' ,'fail' or 'unsure')
    - QA on the trial (Question shouldn't be related to the trial outcome)

#Additional features to extract
- Extract the number of references per trial for each category
- Number of publications after trial completion
- Number of publications before trial completion
'''

from ast import arg
from hmac import new
import json
import glob
from turtle import update
from numpy import save
from torch import ne
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util,models
import os
import time
import pandas as pd
import argparse
from support_functions import extract_study_basic_info,filter_articles,extract_similar_pubmed_articles
import numpy as np



def main(data_path,save_path,dev = False):
    # read extracted pubmed_articles
    pubmed_path = '/'.join(save_path.split('/')[0:-1])
    
    
    
    
    
    # read extracted pubmed articles      
    pubmed_files = glob.glob(os.path.join(pubmed_path,'extracted_pubmed','*_pubmed_abs.json'))
    print(f"Total number of pubmed files: {len(pubmed_files)}")
    embeddings = models.Transformer(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    # Pooling model
    pooling = models.Pooling(embeddings.get_word_embedding_dimension())
    # Create sentence-transformers model
    model = SentenceTransformer(modules=[embeddings, pooling])



    # Extract basic information from the study info file
    trial_basic_info = extract_study_basic_info(os.path.join(data_path,'studies.txt'), info_to_extract = ['official_title','start_date','completion_date'])


    # Extract top 2 similar articles based on trial title and abstract title and create a dataframe
    new_rows = []
    # read previous top 2 extracted pubmed articles csv in llm_prediction folder
    top_2_exists = False
    if os.path.exists(os.path.join(save_path,'top_2_extracted_pubmed_articles.csv')):
        print('Reading previously extracted top 2 similar articles')
        top_2_exists = True
        top_2_prev_df = pd.read_csv(os.path.join(save_path,'top_2_extracted_pubmed_articles.csv'))
        # convert to list of dictionaries
        new_rows = top_2_prev_df.to_dict('records') # changed to list of dictionaries
    
    updated_nct_id = []
    new_nct_id =[]
    # # read data frame and append the rows to new_rows
    # pubmed_df = pd.read_csv('./top_2_extracted_pubmed_articles.csv')
    # print(len(pubmed_df))
    # for i in range(len(pubmed_df)):
    #     new_rows.append(pubmed_df.iloc[i].to_dict())

    # for development mode 
    num = 0
    
    for jsonfile in tqdm(pubmed_files):
        nct_id = jsonfile.split('/')[-1].split('_')[0]
        # if nct_id in pubmed_df['nct_id'].values:
        #     continue
        filtered_articles_list = filter_articles(nct_id, trial_basic_info, pubmed_files)
        if len(filtered_articles_list) == 0:  # if no articles are found for the trial
            continue
        top_2_similar_articles = extract_similar_pubmed_articles(nct_id, trial_basic_info, filtered_articles_list, model)
        row = {
        'nct_id': nct_id,
        'official_title': trial_basic_info[nct_id]['official_title'],
        'start_date': trial_basic_info[nct_id]['start_date'],
        'completion_date': trial_basic_info[nct_id]['completion_date']}
        k = 0
        for article in top_2_similar_articles:
            k+=1
            row[f'top_{k}_similar_article_title'] = article['Title']
            row[f'top_{k}_similar_article_abstract'] = article['Abstract']
            row[f'top_{k}_similar_article_pub_date'] = article['Date of Publication']
            row[f'top_{k}_similar_article_similarity_score'] = article['similarity'].item()
            row[f'top_{k}_similar_article_type'] = article['Reference type']
            row[f'top_{k}_similar_article_journal'] = article['Journal Title']
            row[f'top_{k}_similar_article_PMID'] = article['PMID']
            row[f'top_{k}_similar_article_PMCID'] = article['PMC ID']
            row[f'top_{k}_similar_article_author_affiliation'] = article['Author(s) Affiliation']
            
        # Get counts of the references or pubmed articles per trial
        for json_file in pubmed_files:
            if nct_id not in json_file:
                continue
            with open(json_file, 'r') as f:
                pubmed_all_data = json.load(f)
                f.close()
        
        # # Get article counts for ['background','derived','result']
        # for article_type in ['background','derived','result']:
        #     row[article_type] = 0
        #     for article in pubmed_all_data['References']:
        #         if article['Reference type'].lower() == article_type:
        #             row[article_type] += 1
                    
        # check if the PMID is in the new_rows
        # check if the ncit_id exists in the new_rows
        if top_2_exists:
            nct_id_exists = False
            for i in range(len(new_rows)):
                if new_rows[i]['nct_id'] == nct_id:
                    prev_row = new_rows[i]
                    nct_id_exists = True
                    break
            if nct_id_exists:
                # check if the PMID exists in the new_rows
                prev_pmid_list = []
                for i in range(1,3):
                    prev_pmid_list.append(prev_row[f'top_{i}_similar_article_PMID'])
                #check if top_1_similar_article_PMID and top_2_similar_article_PMID exists in the  row
                try:
                    if row['top_1_similar_article_PMID'] not in prev_pmid_list or row['top_2_similar_article_PMID'] not in prev_pmid_list:
                        
                        # delete the previous row from new_rows and append the new row
                        new_rows.remove(prev_row)
                        new_rows.append(row)
                        updated_nct_id.append(nct_id)
                    else:
                        continue
                except:
                    new_rows.remove(prev_row)
                    new_rows.append(row)
                    updated_nct_id.append(nct_id)
            else:
                new_rows.append(row)
                new_nct_id.append(nct_id)
        else:
            new_rows.append(row)
            new_nct_id.append(nct_id)
            
        num += 1
        # for development mode
        if dev and num == 550:
            print('Development mode: break')
            break
        
        # if len(new_rows) % 10000 == 0:
        #     pubmed_df = pd.DataFrame(new_rows)  
        #     pubmed_df.to_csv('./top_2_extracted_pubmed_articles.csv', index = False)
            
    pubmed_df = pd.DataFrame(new_rows)  
    pubmed_df.to_csv('./top_2_extracted_pubmed_articles.csv', index = False)
    
    # log all updated nct_id with date to log file
    log_path = '/'.join(save_path.split('/')[0:-1])
    if top_2_exists:
        with open(f'{log_path}/logs/pubmed_reference_logs.txt', 'a') as f:
            f.write('====================\n')
            f.write(f'Update time: {time.ctime()}\n')
            f.write('Top 2 similar articles updated for the following nct_ids:\n')
            f.write(f'Updated {len(updated_nct_id)} nct_id: {updated_nct_id}\n')
            f.write('Following nct_ids are new:\n')
            f.write(f'New {len(new_nct_id)} nct_id: {new_nct_id}\n')
            f.close()
        print(f'{time.ctime()}: Updated {len(updated_nct_id)} nct_id: {updated_nct_id}')
        print(f'{time.ctime()}: New {len(new_nct_id)} nct_id: {new_nct_id}')
        print('Top 2 similar articles updated')
    else:
        print('Top 2 similar articles extracted')
        
        
        with open(f'{log_path}/logs/pubmed_reference_logs.txt', 'a') as f:
            f.write('====================\n')
            f.write(f'Update time: {time.ctime()}\n')
            f.write('Top 2 similar articles extracted\n')
            f.write(f'Numeber of nct_ids: {len(new_nct_id)}\n')
            f.close()
        print(f'{time.ctime()}: New {len(new_nct_id)} nct_id: {new_nct_id}')
        print('Top 2 similar articles extracted')
    
    # combine new_nct_id and updated_nct_id and save it as a .npy file
    upd_new_nct_id_list = new_nct_id + updated_nct_id
    
    # save the list as a .npy file
    np.save(f'./updated_new_nct_id.npy', upd_new_nct_id_list)
    # 
#     # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None , help='Path to the CITI data folder')
    # parser.add_argument('--pubmed_path', type=str, default=None , help='Path to the extracted pubmed data')
    parser.add_argument('--save_path', type=str, default= None, help='Path to save the extracted data')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError('Please provide the path to the CITI data folder')
    if args.save_path is None:
        raise ValueError('Please provide the path to main folder where extracted pubmed data is saved')
    
    data_path = args.data_path
    # pubmed_path = args.pubmed_path
    args.save_path = os.path.join(args.save_path,'llm_predictions_on_pubmed')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # os.chdir(args.pubmed_path)
    os.chdir(args.save_path)

    # main(data_path,pubmed_path)
    main(data_path,args.save_path,args.dev)
    
