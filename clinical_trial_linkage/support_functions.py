import os
import json
import pandas as pd
from tqdm import tqdm
import datetime
import re
import openai
import datetime
from sentence_transformers import util


# study info 
def extract_study_basic_info(data_path, info_to_extract = ['official_title','start_date','completion_date']):
    '''
    Extract basic information from the study info file such as
    title, completion data etc. 
    Args:
    - data_path: path to the study info file
    - info_to_extract: list of information to extract ex: ['official_title','start_date','completion_date']
    Returns:    
    - trial_info: Nested dictionary with nct_id as key, extracted informations as sub dictionary
    
    '''
    
    file_data = open(data_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    trial_info = {}
    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        trial_info[nct_id] = {}
        for info in info_to_extract:
            if info == 'official_title' and study[columns.index(info)] == '':
                trial_info[nct_id][info] = study[columns.index('brief_title')]
            elif study[columns.index(info)] == '':
                trial_info[nct_id][info] = ''
            else:
                trial_info[nct_id][info] = study[columns.index(info)]
    return trial_info

def parse_date(date_string):

    # Check and handle month range format (e.g., 2010 May-Jun)
    if '-' in date_string:
        date_string = date_string.split('-', 0)[0]

    try:
        # Try parsing with the format including the day
        return datetime.datetime.strptime(date_string, '%Y %b %d')
    except ValueError:
        # If it fails, try parsing without the day
        try:
            return datetime.datetime.strptime(date_string, '%Y %b')
        except ValueError:
            try:
                # Try parsing with just the year
                return datetime.datetime.strptime(date_string, '%Y')
            except ValueError:
                # As a last resort, try extracting the year using a regular expression
                year_match = re.search(r'20\d{2}|19\d{2}', date_string)
                if year_match:
                    year = int(year_match.group())
                    return datetime.datetime(year, 1, 1)
                else:
                    # If all attempts fail, re-raise the error
                    raise ValueError("Date string does not match expected formats ('%Y %b %d', '%Y %b', or '%Y')")


def filter_articles(nct_id, trial_basic_info, pubmed_files):
    '''
    Filter articles based on completion date + year and type of article
    Args:
    - nct_id: NCT ID of the trial
    - trial_basic_info: Nested dictionary with nct_id as key, extracted information as sub dictionary
    - pubmed_files: list of file names of jsons. Each json is a dictionary with the List of sub dictionary of information on pubmed files.
    Returns:
    - filtered_articles: List of articles that are derived and results articles and are within the completion date of the trial
    '''
    filtered_articles = []
    datetime_format = "%Y-%m-%d"
    for json_file in pubmed_files:
        if nct_id not in json_file:
            continue
        # print(nct_id, json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
            f.close()
        if data['nct_id'] == nct_id:
            reference_list = data['References']
            
            # if completion date is not available, append all references
            if trial_basic_info[nct_id]['completion_date']is None or trial_basic_info[nct_id]['completion_date'] == '':
                for reference in reference_list:
                    if reference['Reference type'] in ['derived','result']:
                        filtered_articles.append(reference)
                continue
            
            for reference in reference_list:
                if reference['Reference type'] in ['derived','result']:
                    # if reference['Date of Publication'] is not available, append the reference
                    if reference['Date of Publication'] == '':
                        filtered_articles.append(reference)
                        continue
                    trial_completion_date = trial_basic_info[nct_id]['completion_date']
                    trial_completion_date = datetime.datetime.strptime(trial_completion_date, datetime_format)
                    publication_date = reference['Date of Publication']
                    publication_date = parse_date(publication_date)
                    # print(trial_completion_date, publication_date)
                    # publication date should be less than a year after the trial completion date
                    if publication_date <= trial_completion_date + datetime.timedelta(days=5*365):
                        filtered_articles.append(reference)
            if len(filtered_articles) == 0:
                # print('No articles found for trial before 5 years after completion:', nct_id)
                # print('appending all articles')
                for reference in reference_list:
                    if reference['Reference type'] in ['derived','result']:
                        filtered_articles.append(reference)
                
    return filtered_articles

#function to extract top 2 similar articles based on trial title and abstract title
def extract_similar_pubmed_articles(nct_id, trial_basic_info, filtered_articles_list, model):
    '''
    Extract top 2 similar articles based on trial title and abstract title
    Args:
    - nct_id: NCT ID of the trial
    - trial_basic_info: Nested dictionary with nct_id as key, extracted information as sub dictionary
    - filtered_articles_list: List of articles that are derived and results articles and are within the completion date of the trial
    - model: Sentence transformer model
    Returns:
    - top_2_similar_articles: List of top 2 similar articles based on trial title and abstract title
    '''
    top_2_similar_articles = []
    trial_title = trial_basic_info[nct_id]['official_title']
    trial_title_embedding = model.encode(trial_title)
    for article in filtered_articles_list:
        article_title = article['Title']
        article_title_embedding = model.encode(article_title)
        similarity_title = util.cos_sim(trial_title_embedding, article_title_embedding)
        article['similarity'] = similarity_title
        top_2_similar_articles.append(article)
        
    top_2_similar_articles = sorted(top_2_similar_articles, key = lambda x: x['similarity'], reverse = True)[:2]
    return top_2_similar_articles

# For azure
def init_api(api_key, azure_endpoint, api_version):
    '''
    Initialize the AzureOpenAI API and set the environment variables
    '''

    os.environ["TOKENIZERS_PARALLELISM"]='false' 
    os.environ["OPENAI_API_KEY"] =  api_key
    os.environ["AZURE_OPENAI_API_KEY"] =  api_key
    os.environ["OPENAI_API_BASE"] = azure_endpoint
    os.environ["OPENAI_API_VERSION"] = api_version
    os.environ["OPENAI_API_TYPE"] = "azure"


    openai.api_type = "azure"
    openai.api_base = azure_endpoint
    openai.api_version = api_version
    openai.api_key = api_key
    openai.end_point = azure_endpoint