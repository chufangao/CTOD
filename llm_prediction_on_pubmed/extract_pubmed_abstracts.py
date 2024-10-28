from operator import is_
from tkinter import N
from Bio import Medline
import urllib.request as urllib
import json
from torch import le
from tqdm import tqdm
import os
import time
import argparse
from utils import drug_biologics_nct_ids
import pandas as pd



def get_data(element, source):
        """."""
        value = source.get(element, "")
        if isinstance(value, list):
            value = '||'.join(value)
        return value

def get_all_data(article):
    """."""
    article_data = {}
    article_data["PMID"] = get_data("PMID", article)
    article_data["PMC ID"] = get_data("PMC", article)
    article_data["Title"] = get_data("TI", article)
    article_data["Author(s) Affiliation"] = get_data("AD", article)
    article_data["Collaborator(s) Affiliation"] = get_data("IRAD", article)
    article_data["Journal Title Abbreviation"] = get_data("TA", article)
    article_data["Journal Title"] = get_data("JT", article)
    abstract = get_data("AB", article)
    if not abstract:
        abstract = ' '.join(article.get("OAB", ""))
    article_data["Abstract"] = abstract
    copyright = get_data("CI", article)
    if not copyright:
        copyright = get_data("OCI", article)
    article_data["Copyright Information"] = copyright
    article_data["Grant Number"] = get_data("GR", article)
    article_data["Date of Publication"] = get_data("DP", article)
    article_data["Date of Electronic Publication"] = get_data("DEP", article)
    article_data["Corrected and Republished in"] = get_data("CRI", article)
    article_data["Corrected and Republished from"] = get_data("CRF", article)
    article_data["Owner"] = get_data("OWN", article)
    return article_data


def main(data_path,NCBI_api_key, dev = False):
    print('Extracting data from study_references.txt')
    study_ref_path = os.path.join(data_path,'study_references.txt')
    
    # study_ref_file = open(study_ref_path, "r").read().split('\n')
    study_ref_df = pd.read_csv(study_ref_path, sep='|')
    study_ref_df = study_ref_df.dropna(subset=['pmid'])
    
    # filter the nct_id that are drug or biological
    intervention_path = os.path.join(data_path,'interventions.txt')
    drug_biologics_nct_ids_list = drug_biologics_nct_ids(intervention_path)
    study_ref_df = study_ref_df[study_ref_df['nct_id'].isin(drug_biologics_nct_ids_list)].reset_index(drop=True)
    print(f'Number of nct_id with drug or biological intervention: {len(study_ref_df)}')
    study_ref_dict = study_ref_df.to_dict()                                 
    
    
    from collections import defaultdict
    nct_id_dict = defaultdict(list) # create a dictionary with empty lists as values
    for i in range(len(study_ref_df)):
        nct_id_dict[study_ref_df['nct_id'][i]].append([study_ref_df['pmid'][i],study_ref_df['reference_type'][i]])
        # nct_id_dict[study_ref_file[i].split('|')[1]].append(study_ref_file[i].split('|')[2:])
    print('Number of nct_id with references:', len(nct_id_dict))



    MEDLINE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
    MEDLINE_URL = MEDLINE_URL + "&api_key=" + NCBI_api_key
    MEDLINE_URL = MEDLINE_URL + "&rettype=medline"
    MEDLINE_TEXT_URL = MEDLINE_URL + "&retmode=text&id="

    


    print('start')
    if not os.path.exists('./extracted_pubmed'):
        os.makedirs('./extracted_pubmed')
    
    num = 0
    total = len(nct_id_dict)
    missed_nct_id = []
    updated_nct_id = []
    for k,v in tqdm(nct_id_dict.items()):
        
        try:
        
            nct_id = k
            
            
            reference_data = {}
            reference_data['nct_id'] = nct_id
            
            reference_list = []
            #check whether the file exists
            trial_ref_exists_in_data = False
            if os.path.exists(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json')): # checking if the trial references scraped previously
                existing_reference_dict = json.load(open(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json')))
                reference_list = existing_reference_dict['References']
                #get PMID from existing references
                existing_pmids = [reference_list[i]['PMID'] for i in range(len(reference_list))]
                trial_ref_exists_in_data = True
                # continue
            
            is_file_updated = False
            for i in range(len(v)):
                pmid= v[i][0]
                if trial_ref_exists_in_data: # if the trial reference exists in the data, skip the reference
                    if pmid in existing_pmids:
                        continue
                is_file_updated = True  
                ref_type = v[i][1]
                # ignore background references
                if ref_type.lower() == 'background':
                    continue 
                # print(nct_id, pmid, ref_type)
                text_path = './pubmed_data.txt'
                urllib.urlretrieve(MEDLINE_TEXT_URL + str(pmid), text_path)
                with open(text_path, mode="r", encoding="utf-8") as handle:
                    articles = Medline.parse(handle)
                    for article in articles:
                        article_data = get_all_data(article)
                        article_data['Reference type'] = ref_type
                        reference_list.append(article_data)
                    handle.close()
                
            reference_data['References'] = reference_list
            with open(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json'), 'w') as f:
                json.dump(reference_data, f)
            if is_file_updated:
                updated_nct_id.append(k)
        except:
            missed_nct_id.append(k)
            print(f'Error with {k}')
            time.sleep(5)
            num += 1
            continue
        
        # for development mode
        num += 1
        if dev and num == 550:
            print('Development mode: break')
            break

    # log all updated nct_id with date to log file
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    with open('./logs/pubmed_reference_logs.txt', 'a') as f:
        f.write('====================\n')
        f.write(f'Update time: {time.ctime()}\n')
        f.write('Extracting pubmed abstracts\n')
        f.write(f'Updated {len(updated_nct_id)} nct_id: {updated_nct_id}\n')
        f.close()
    print(f'{time.ctime()} - Updated {len(updated_nct_id)} nct_id: {updated_nct_id}')
    print('Pubmed abstracts extraction completed')
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= None, help='Path to the CITI data folder')
    parser.add_argument('--NCBI_api_key', type=str, default= None, help='NCBI API key')
    parser.add_argument('--save_path', type=str, default= None, help='Path to save the extracted data')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    data_path = args.data_path
    NCBI_api_key = args.NCBI_api_key
    
    if data_path is None:
        raise ValueError('Please provide the path to the CITI data folder')
    if NCBI_api_key is None:
        raise ValueError('Please provide the NCBI API key')
    if args.save_path is None:
        raise ValueError('Please provide the path to save the extracted data')
    
    # change to path to save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # args.save_path = os.path.join(args.save_path,'llm_predictions_on_pubmed')
    os.chdir(args.save_path)
    
    
    print('Extracting PubMed abstracts')
    main(data_path,NCBI_api_key, args.dev)
    print('Done')
    

