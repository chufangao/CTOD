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

import json
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util,models
import os
import pandas as pd
import argparse
from support_functions import extract_study_basic_info,filter_articles,extract_similar_pubmed_articles




def main(data_path,pubmed_path):
    # read extracted pubmed_articles
    
    
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
    # # read data frame and append the rows to new_rows
    # pubmed_df = pd.read_csv('./top_2_extracted_pubmed_articles.csv')
    # print(len(pubmed_df))
    # for i in range(len(pubmed_df)):
    #     new_rows.append(pubmed_df.iloc[i].to_dict())


    for jsonfile in tqdm(pubmed_files):
        nct_id = jsonfile.split('/')[-1].split('_')[0]
        # if nct_id in pubmed_df['nct_id'].values:
        #     continue
        
        filtered_articles_list = filter_articles(nct_id, trial_basic_info, pubmed_files)
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
        
        # Get article counts for ['background','derived','result']
        for article_type in ['background','derived','result']:
            row[article_type] = 0
            for article in pubmed_all_data['References']:
                if article['Reference type'] == article_type:
                    row[article_type] += 1
        new_rows.append(row)
        
        if len(new_rows) % 10000 == 0:
            pubmed_df = pd.DataFrame(new_rows)  
            pubmed_df.to_csv('./top_2_extracted_pubmed_articles.csv', index = False)
            
    pubmed_df = pd.DataFrame(new_rows)  
    pubmed_df.to_csv('./top_2_extracted_pubmed_articles.csv', index = False)
#     # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None , help='Path to the CITI data folder')
    parser.add_argument('--pubmed_path', type=str, default=None , help='Path to the extracted pubmed data')
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError('Please provide the path to the CITI data folder')
    if args.pubmed_path is None:
        raise ValueError('Please provide the path to the extracted pubmed data')
    
    data_path = args.data_path
    pubmed_path = args.pubmed_path
    
    os.chdir(args.pubmed_path)

    main(data_path,pubmed_path)
    
# python retrieve_top2_abstracts.py --data_path /home/jp65/CTOD/data --pubmed_path /home/jp65/CTOD