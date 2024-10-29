from Bio import Medline
import urllib.request as urllib
import json
from tqdm import tqdm
import os
import time
import argparse
from utils import drug_biologics_nct_ids
import pandas as pd
from dotenv import load_dotenv
from Bio import Entrez



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

def get_pmids(query,max_results = 5,email = ''):
    Entrez.email = email
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax=str(max_results),
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    
    pmids = set(results['IdList'])
    return pmids


def search_and_extract_pubmed(data_path,NCBI_api_key,email = ''):
    
    # filter the nct_id that are drug or biological
    intervention_path = os.path.join(data_path,'interventions.txt')
    drug_biologics_nct_ids_list = drug_biologics_nct_ids(intervention_path)
    drug_biologics_nct_ids_list =set(drug_biologics_nct_ids_list)
    
    MEDLINE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
    MEDLINE_URL = MEDLINE_URL + "&api_key=" + NCBI_api_key
    MEDLINE_URL = MEDLINE_URL + "&rettype=medline"
    MEDLINE_TEXT_URL = MEDLINE_URL + "&retmode=text&id="

    
    updated_nct_id = []
    for nct_id in tqdm(drug_biologics_nct_ids_list):
        
        try:
            pmids = get_pmids(nct_id,max_results = 5,email = email)
            pmids = list(pmids)
            trial_ref_exists_in_data = False
            is_file_updated = False
            reference_data = {}
            reference_data['nct_id'] = nct_id
            reference_list = []
            if os.path.exists(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json')):
                existing_reference_dict = json.load(open(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json')))
                reference_list = existing_reference_dict['References']
                #get PMID from existing references
                existing_pmids = [reference_list[i]['PMID'] for i in range(len(reference_list))]
                existing_reference_types = [reference_list[i]['Reference type'].lower() for i in range(len(reference_list))]
                if 'result' in existing_reference_types or 'search_result' in existing_reference_types:
                    continue
                trial_ref_exists_in_data = True
            
            if trial_ref_exists_in_data:
                # filter out existing pmids
                pmids = [pmid for pmid in pmids if pmid not in existing_pmids]
                
            if len(pmids) == 0:
                continue
            else:
                is_file_updated = True
                for pmid in pmids:
                    text_path = './pubmed_data.txt'
                    urllib.urlretrieve(MEDLINE_TEXT_URL + str(pmid), text_path)
                    with open(text_path, mode="r", encoding="utf-8") as handle:
                        articles = Medline.parse(handle)
                        for article in articles:
                            article_data = get_all_data(article)
                            article_data['Reference type'] = 'search_result'
                            reference_list.append(article_data)
                        handle.close()
            reference_data['References'] = reference_list  
            with open(os.path.join('./extracted_pubmed',f'{nct_id}_pubmed_abs.json'), 'w') as f:
                json.dump(reference_data, f)
            if is_file_updated:
                updated_nct_id.append(nct_id)
        except:
            print(f'Error with {nct_id}')
            time.sleep(5)
            continue
        
    
    # log all updated nct_id with date to log file
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    with open('./logs/pubmed_search_logs.txt', 'a') as f:
        f.write('====================\n')
        f.write(f'Update time: {time.ctime()}\n')
        f.write('Extracting pubmed abstracts\n')
        f.write(f'Updated {len(updated_nct_id)} nct_id: {updated_nct_id}\n')
        f.close()
    print(f'{time.ctime()} - Updated {len(updated_nct_id)} nct_id: {updated_nct_id}')
    print('Pubmed abstracts search extraction completed')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= None, help='Path to the CITI data folder')
    # parser.add_argument('--NCBI_api_key', type=str, default= None, help='NCBI API key')
    parser.add_argument('--save_path', type=str, default= None, help='Path to save the extracted data')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    args = parser.parse_args()
    
    data_path = args.data_path
    load_dotenv()
    NCBI_api_key = os.getenv('NCBI_api_key')
    email = os.getenv('NCBI_email')
    
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
    
    
    print('Search and Extracting PubMed abstracts')
    search_and_extract_pubmed(data_path,NCBI_api_key,email = email)
    print('Done')        
    
            
