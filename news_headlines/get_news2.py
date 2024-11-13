import os
import sys
from tqdm import tqdm, trange
from datetime import datetime, timedelta
import time
import torch
import numpy as np
from transformers import pipeline
import pandas as pd
import random
import json
import argparse
from collections import Counter
# # append GNews to path, append the path to the GNews folder, in this case, the GNews folder is in the directory of the script
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'GNews/'))

def gnews_search(query):
    from gnews import GNews
    google_news = GNews()
    news = google_news.get_news(query)
    return news

def request_search(query):
    from bs4 import BeautifulSoup
    import requests
    install_folder = os.path.abspath(os.path.split(__file__)[0])
    with open(os.path.join(install_folder, "useragents.txt"), 'r') as f:
        USERAGENT_LIST = [_.strip() for _ in f.readlines()]
    headers = {'User-Agent': random.choice(USERAGENT_LIST)}
    response = requests.get(f"https://www.google.com/search?q={query}&gl=us&tbm=nws&num=100", headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []
    for el in soup.select("div.SoaBEf"):
        news_results.append({
                "link": el.find("a")["href"],
                "title": el.select_one("div.MBeuO").get_text(),
                "snippet": el.select_one(".GI74Re").get_text(),
                "date": el.select_one(".LfVVr").get_text(),
                "source": el.select_one(".NUnG9d span").get_text()
            })
    # print(json.dumps(news_results, indent=2))
    return news_results

def yagoogle_search(query):
    import yagooglesearch
    client = yagooglesearch.SearchClient(
        query,
        tbs="li:1",
        max_search_result_urls_to_return=100,
        http_429_cool_off_time_in_minutes=60,
        http_429_cool_off_factor=1.5,
        # verbosity=5,
        verbose_output=True,  # False (only URLs) or True (rank, title, description, and URL)
    )
    client.assign_random_user_agent()
    urls = client.search()
    return urls

def serpapi_search(query):
    import serpapi
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    results = client.search({
        'engine': 'google',
        "tbm": "nws",
        'q': query
    })
    return results.as_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='get_news', help='get_news, process_news, correspond_news_and_studies')
    parser.add_argument('--continue_from_prev_log', type=bool, default=False)
    parser.add_argument('--CTTI_PATH', type=str, default='./CITT/')
    parser.add_argument('--SENTIMENT_MODEL', type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # parser.add_argument('--SENTENCE_ENCODER', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    # parser.add_argument('--SENTENCE_CROSSENCODER', type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument('--SAVE_NEWS_LOG_PATH', type=str, default='./news_logs/')
    # parser.add_argument('--SAVE_NEWS_EMBEDDING_PATH', type=str, default='./news_title_embeddings.npy')
    # parser.add_argument('--SAVE_STUDY_TITLE_EMBEDDING_PATH', type=str, default='./studies_title2_embeddings.npy')
    # parser.add_argument('--SAVE_NEWS_PATH', type=str, default='./news.csv')
    parser.add_argument('--SAVE_STUDY_NEWS_PATH', type=str, default='/srv/local/data/CTO/new_headlines/news_lfs.csv')
    parser.add_argument('--NCT_IDS_TO_PROCESS', type=str, default=None)
    args = parser.parse_args()
    print(args)
    assert args.mode in ['get_news', 'process_news', 'correspond_news_and_studies']

    print(f'args.mode: {args.mode}')
    print('Loading CTTI data')
    studies = pd.read_csv(args.CTTI_PATH + 'studies.txt.zip', sep='|', low_memory=False)
    interventions = pd.read_csv(args.CTTI_PATH + 'interventions.txt.zip', sep='|', low_memory=False)
    print('Loaded CTTI data')
    
    interventions = interventions[interventions['intervention_type'].isin(['DRUG', 'BIOLOGICAL'])]
    studies = studies[studies['nct_id'].isin(interventions['nct_id'])]
    studies = studies[studies['overall_status']=='COMPLETED']
    studies.dropna(subset=['phase'], inplace=True)
    studies = studies[studies['phase'].str.contains('1') | studies['phase'].str.contains('2') | studies['phase'].str.contains('3')]

    if args.NCT_IDS_TO_PROCESS is not None:
        nct_ids_to_process = pd.read_csv(args.NCT_IDS_TO_PROCESS)
        studies = studies[studies['nct_id'].isin(nct_ids_to_process['nct_id'])]
    print(f'Processing {studies.shape[0]} studies')
    
    # studies = studies.iloc[30000:]
    if not args.continue_from_prev_log:
        studies = studies.iloc[:len(studies)//2]
    else:
        studies = studies.iloc[len(studies)//2:]

    if args.mode == 'get_news': # warning: this will take a long time (multiple weeks)
        for nct_id in tqdm(studies['nct_id']):
            # print(f'Getting news for {nct_id}')
            if os.path.exists(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json")):
                with open(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json"), 'rb') as f:
                    news = json.load(f)
                if len(news) > 0:
                    print(f'Loaded {len(news)} news for {nct_id}')
                    continue
                
            # time.sleep(random.random() * 1/2)
            # news = gnews_search(f"{nct_id} clinical trial")
            # news = yagoogle_search(f"{nct_id}")
            news = serpapi_search(f"{nct_id} clinical trial")
            if type(news) is dict:
                if 'news_results' in news:
                    print(f'Got {len(news["news_results"])} news for {nct_id}')
            elif type(news) is list:                
                if len(news) > 0:
                    print(f'Got {len(news)} news for {nct_id}')
                
            with open(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json"), "w") as f:
                json.dump(news, f)

    if args.mode == 'process_news': # warning: this will take a long time (multiple weeks)

        # from sentence_transformers import SentenceTransformer

        # encoder = SentenceTransformer('all-mpnet-base-v2')

        device = 'cpu'

        # sentiment_model = pipeline("text-classification", model=SENTIMENT_MODEL, device=device)
        sentiment_task = pipeline("sentiment-analysis", model=args.SENTIMENT_MODEL)
        # sentiment_task("Covid cases are increasing fast!")

        # positive_embed = encoder.encode('clinically significant, positive')
        # negative_embed = encoder.encode('negative failed terminated unsuccessful adverse')
        # print(positive_embed.shape, negative_embed.shape)

        news_dict = {}
        for file in tqdm(os.listdir(args.SAVE_NEWS_LOG_PATH)):
            with open(os.path.join(args.SAVE_NEWS_LOG_PATH, file)) as f:
                data = json.load(f)
            # print(file, len(data))
            if 'news_results' in data.keys():
                for i in range(len(data['news_results'])):
                        #         out = sentiment_model(titles[i:i+batch_size], batch_size=batch_size)
            #         all_label_preds += [o['label'] for o in out]
            #         all_label_probs += [o['score'] for o in out]
            #     all_company_dfs['sentiment'] = all_label_preds
            #     all_company_dfs['sentiment_prob'] = all_label_probs
                    # out = encoder.encode(data['news_results'][i]['title'] + ' ' + data['news_results'][i]['snippet'])
                    out = sentiment_task(data['news_results'][i]['title'] + ' ' + data['news_results'][i]['snippet'])
                    data['news_results'][i]['sentiment'] = out[0]['label']
                    data['news_results'][i]['sentiment_prob'] = out[0]['score']
                news_dict[file.replace('.json','')] = data['news_results']

        news_df = pd.DataFrame.from_dict(news_dict, orient='index').reset_index().rename(columns={'index': 'nct_id'})
        news_cols = news_df.columns[1:]
        # news_df['valid_sentiments'] = news_df.apply(lambda x: np.mean([x[i]['sentiment'] for i in news_cols if x[i] is not None]), axis=1)
        news_df['valid_sentiments'] = news_df.apply(lambda x: [x[i]['sentiment'] for i in news_cols if x[i] is not None], axis=1)
        # mode
        news_df['mode'] = news_df['valid_sentiments'].apply(lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else 'None')

        # news_df['mode']

        news_df['lf'] = -1
        news_df.loc[news_df['mode']=='positive', 'lf'] = 1
        news_df.loc[news_df['mode']=='neutral', 'lf'] = 1
        news_df.loc[news_df['mode']=='negative', 'lf'] = 0

        print(news_df['lf'].value_counts())

        news_df.to_csv(args.SAVE_STUDY_NEWS_PATH, index=False)        
    else:
        raise NotImplementedError
    # # ======================== Process the news data ========================
    # encoder = SentenceTransformer(args.SENTENCE_ENCODER)
    # crossencoder = CrossEncoder(args.SENTENCE_CROSSENCODER, max_length=512)
                
    # elif args.mode == 'process_news':
    #     print('Processing news data')
    # from transformers import pipeline
    # from sentence_transformers import SentenceTransformer, CrossEncoder

    #     all_company_dfs = []
    #     for company in sorted(os.listdir(args.SAVE_NEWS_LOG_PATH)):
    #         with open(os.path.join(company), 'rb') as f:
    #             news = json.load(f)
    #         # print(company, ticker, news)

    #         all_titles = []
    #         all_descriptions = []
    #         all_dates = []
    #         all_publishers = []
    #         for k in news.keys():
    #             if len(news[k]) > 0:
    #                 # print(k, news[k][0].keys())
    #                 for i in range(len(news[k])):
    #                     date = convert_to_datetime(news[k][i]['published date'])
    #                     # print(news[k][i]['published date'],date); quit()
    #                     all_dates.append(date)
    #                     all_titles.append(news[k][i]['title'])
    #                     all_descriptions.append(news[k][i]['description'])
    #                     all_publishers.append(news[k][i]['publisher']['title'])

    #         df = pd.DataFrame({'date': all_dates, 'title': all_titles, 'description': all_descriptions, 'publisher': all_publishers})
    #         df['company'] = company
    #         all_company_dfs.append(df)
    #     all_company_dfs = pd.concat(all_company_dfs)

    #     print("Processing sentiment")
    #     batch_size = 512
    #     all_label_preds = []
    #     all_label_probs = []
    #     titles = all_company_dfs['title'].tolist()
    #     for i in trange(0, len(titles), batch_size):
    #         out = sentiment_model(titles[i:i+batch_size], batch_size=batch_size)
    #         all_label_preds += [o['label'] for o in out]
    #         all_label_probs += [o['score'] for o in out]
    #     all_company_dfs['sentiment'] = all_label_preds
    #     all_company_dfs['sentiment_prob'] = all_label_probs

    #     # process title embeddings using pubmedbert
    #     print("Processing embeddings")
    #     encoded_titles = encoder.encode(titles, convert_to_numpy=True, device=device, batch_size=batch_size)
    #     np.save(args.SAVE_NEWS_EMBEDDING_PATH, encoded_titles)

    #     all_company_dfs.to_csv(args.SAVE_NEWS_PATH, index=False)

    # elif args.mode == 'correspond_news_and_studies':
        # from transformers import pipeline
        # from sentence_transformers import SentenceTransformer, CrossEncoder                
    #     news_df = pd.read_csv(args.SAVE_NEWS_PATH)
    #     news_title_embedding = np.load(args.SAVE_NEWS_EMBEDDING_PATH)
    #     top_sponsors = combined
    #     interventions = pd.read_csv(args.CTTI_PATH+'interventions.txt.zip', sep='|')
    #     conditions = pd.read_csv(args.CTTI_PATH+'conditions.txt.zip', sep='|')

    #     studies = studies[studies['nct_id'].isin(top_sponsors['nct_id'])]
    #     studies = studies[studies['nct_id'].isin(interventions['nct_id'])]
    #     studies = studies[studies['nct_id'].isin(conditions['nct_id'])]
    #     news_df['date'] = pd.to_datetime(news_df['date'])
    #     studies['completion_date'] = pd.to_datetime(studies['completion_date'])
    #     # # studies['intervention_name'] = studies['nct_id'].map(interventions.set_index('nct_id')['name'])
    #     # studies_title_embedding = studies_title_embedding[studies.index.values]

    #     # map intervention and condition to study name2
    #     interventions['name'] = interventions['name'].astype(str)
    #     intervention_names = interventions.groupby('nct_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
    #     intervention_names.columns = ['nct_id', 'intervention_name']
    #     condition_names = conditions.groupby('nct_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
    #     condition_names.columns = ['nct_id', 'condition_name']
    #     studies = pd.merge(studies, intervention_names, on='nct_id', how='left')
    #     studies = pd.merge(studies, condition_names, on='nct_id', how='left')
    #     studies['title2'] = studies['intervention_name'] + ' ' + studies['condition_name']

    #     studies_title2_embedding = encoder.encode(studies['title2'], convert_to_numpy=True, device='cuda', show_progress_bar=True)
    #     np.save(args.SAVE_STUDY_TITLE_EMBEDDING_PATH, studies_title2_embedding)

    #     # print(news_df.shape, news_title_embedding.shape, studies.shape, studies_title2_embedding.shape)
    #     # # most relevant news for each study

    #     # for each study, filter out news that are not within 2 year of completion date
    #     top_k = 10
    #     topk_cols = [f'top_{i}' for i in range(top_k, 0, -1)] + [f'top_{i}_sim' for i in range(top_k, 0, -1)]
    #     studies[topk_cols] = pd.NA
    #     column_ind = studies.columns.get_loc(f'top_{top_k}')

    #     for i in trange(studies.shape[0]):
    #         if studies.iloc[i]['completion_date'] is pd.NaT:
    #             continue
    #         # get sponsors of the study
    #         sponsors_ = top_sponsors[top_sponsors['nct_id'] == studies.iloc[i]['nct_id']]['name'].tolist()
    #         news_df_ = news_df[news_df['company'].isin(sponsors_)]
    #         news_df_ = news_df_[
    #             np.abs((news_df_['date'] - studies.iloc[i]['completion_date']).dt.days) < 365*2
    #             ]
    #         # print(news_df_.shape)
    #         if news_df_.shape[0] == 0:
    #             continue
    #         news_title_embedding_ = news_title_embedding[news_df_.index]
            
    #         similarity = studies_title2_embedding[i] @ news_title_embedding_.T
    #         inds = np.argsort(similarity)[-top_k:]
    #         sims = crossencoder.predict([(studies.iloc[i]['title2'], news_df_.iloc[ind]['title']) for ind in inds], show_progress_bar=False)

    #         # original inds are indices in news_df_, we need to convert them to indices in news_df
    #         studies.iloc[i, column_ind:column_ind+len(inds)] = news_df_.iloc[inds].index
    #         studies.iloc[i, column_ind+top_k:column_ind+top_k+len(news_df_)] = sims

    #     studies.to_csv(args.SAVE_STUDY_NEWS_PATH, index=False)
