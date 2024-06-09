from Bio import Medline
import urllib.request as urllib
import json
from tqdm import tqdm
import os
import time
print('extract_pubmed_abstracts.py')


# read txt file and print out the content
study_ref_path = '/home/jp65/CTOD/data/study_references.txt'

study_ref_file = open(study_ref_path, "r").read().split('\n')
# study_ref_file[-1]


# extract id nct_id, pmid, reference_type and citation and store in a dictionary
study_ref_dict = {'id':{},'nct_id':{},'pmid':{},'reference_type':{},'citation':{}} # create a dictionary with empty lists as values
for i in range(1,len(study_ref_file)-1):
    study_ref_dict['id'][i-1] = study_ref_file[i].split('|')[0]
    study_ref_dict['nct_id'][i-1] = study_ref_file[i].split('|')[1]
    study_ref_dict['pmid'][i-1] = study_ref_file[i].split('|')[2]
    study_ref_dict['reference_type'][i-1] = study_ref_file[i].split('|')[3]
    study_ref_dict['citation'][i-1] = study_ref_file[i].split('|')[4]


from collections import defaultdict
nct_id_dict = defaultdict(list) # create a dictionary with empty lists as values
for i in range(1,len(study_ref_file)-1):
    nct_id_dict[study_ref_file[i].split('|')[1]].append(study_ref_file[i].split('|')[2:])
len(nct_id_dict)



NCBI_api_key = '558a8ec64b0df1607941d0261d0a5d273308'
MEDLINE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
MEDLINE_URL = MEDLINE_URL + "&api_key=" + NCBI_api_key
MEDLINE_URL = MEDLINE_URL + "&rettype=medline"
MEDLINE_TEXT_URL = MEDLINE_URL + "&retmode=text&id="

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


print('start')

num = 0
total = len(nct_id_dict)
missed_nct_id = []
for k,v in tqdm(nct_id_dict.items()):
    
    try:
    #print an update to show progress as a bar for every 10% of the total
    # if num % (total/100) == 0:
    #     print(num//total)
    # num += 1
    
        nct_id = k
        reference_data = {}
        reference_data['nct_id'] = nct_id
        
        reference_list = []
        #check whether the file exists
        if os.path.exists(f'/home/jp65/CTOD/extracted_pubmed/{nct_id}_pubmed_abs.json'):
            continue
        
        for i in range(len(v)):
            pmid= v[i][0]
            ref_type = v[i][1]  
            # print(nct_id, pmid, ref_type)
            text_path = '/home/jp65/CTOD/pubmed_data.txt'
            urllib.urlretrieve(MEDLINE_TEXT_URL + str(pmid), text_path)
            with open(text_path, mode="r", encoding="utf-8") as handle:
                articles = Medline.parse(handle)
                for article in articles:
                    article_data = get_all_data(article)
                    article_data['Reference type'] = ref_type
                    reference_list.append(article_data)
                handle.close()
            
        reference_data['References'] = reference_list
        with open(f'/home/jp65/CTOD/extracted_pubmed/{nct_id}_pubmed_abs.json', 'w') as f:
            json.dump(reference_data, f)
    except:
        missed_nct_id.append(k)
        print(f'Error with {k}')
        time.sleep(5)
        continue
        
    
    # break

# reference_data

#save as json

