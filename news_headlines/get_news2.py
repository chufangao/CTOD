import os
import sys
from tqdm import tqdm, trange
from datetime import datetime, timedelta
import time
import pandas as pd
import random
import json
import argparse
from gnews import GNews
# # append GNews to path, append the path to the GNews folder, in this case, the GNews folder is in the directory of the script
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'GNews/'))
from bs4 import BeautifulSoup
import requests
import json
import random

USERAGENT_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'
]

def getNewsData(query):
# getNewsData('NCT04994509 clinical trial')
    headers = {
        'User-Agent': random.choice(USERAGENT_LIST)
    }
    response = requests.get(f"https://www.google.com/search?q={query}&gl=us&tbm=nws&num=100",
                            headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []

    for el in soup.select("div.SoaBEf"):
        news_results.append(
            {
                "link": el.find("a")["href"],
                "title": el.select_one("div.MBeuO").get_text(),
                "snippet": el.select_one(".GI74Re").get_text(),
                "date": el.select_one(".LfVVr").get_text(),
                "source": el.select_one(".NUnG9d span").get_text()
            }
        )

    # print(json.dumps(news_results, indent=2))
    return news_results


# def get_top_sponsors(sponsors, studies):
#     """
#     Get the top 1000 most popular phase 3 industry sponsors
    
#     sponsors: pd.DataFrame, sponsors.txt.zip
#     studies: pd.DataFrame, studies.txt.zip

#     Returns: pd.DataFrame, top 1000 most popular phase 3 industry sponsors
#     """
#     # sponsors = pd.read_csv(args.CTTI_PATH + './CTTI/sponsors.txt.zip', sep='|')
#     # studies = pd.read_csv(args.CTTI_PATH + './CTTI/studies.txt.zip', sep='|', low_memory=False)
#     studies['study_first_submitted_date'] = pd.to_datetime(studies['study_first_submitted_date'])
#     sponsors = pd.merge(sponsors, studies[['nct_id', 'phase', 'study_first_submitted_date']], on='nct_id', how='left')
#     sponsors = sponsors[sponsors['agency_class']=='INDUSTRY']
#     sponsors.dropna(inplace=True)
#     sponsors = sponsors[sponsors['phase'].str.contains('Phase 3')]
#     top_sponsors = sponsors['name'].value_counts().head(1000)
#     # coverage_ = top_sponsors.sum() / sponsors['name'].value_counts().sum()
#     # print(coverage_) # 0.8548555767913166
#     combined = pd.merge(top_sponsors.reset_index(),
#                         sponsors.groupby('name')['study_first_submitted_date'].min().reset_index(),
#                         on='name', how='left')
#     # return combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='get_news', help='get_news, process_news, correspond_news_and_studies')
    parser.add_argument('--continue_from_prev_log', type=bool, default=False)
    parser.add_argument('--CTTI_PATH', type=str, default='./CITT/')
    parser.add_argument('--SENTIMENT_MODEL', type=str, default="yiyanghkust/finbert-tone")
    parser.add_argument('--SENTENCE_ENCODER', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--SENTENCE_CROSSENCODER', type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument('--SAVE_NEWS_LOG_PATH', type=str, default='./news_logs/')
    parser.add_argument('--SAVE_NEWS_EMBEDDING_PATH', type=str, default='./news_title_embeddings.npy')
    parser.add_argument('--SAVE_STUDY_TITLE_EMBEDDING_PATH', type=str, default='./studies_title2_embeddings.npy')
    parser.add_argument('--SAVE_NEWS_PATH', type=str, default='./news.csv')
    parser.add_argument('--SAVE_STUDY_NEWS_PATH', type=str, default='./studies_with_news.csv')
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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # sentiment_model = pipeline("text-classification", model=args.SENTIMENT_MODEL, device=device)
    # encoder = SentenceTransformer(args.SENTENCE_ENCODER)
    # crossencoder = CrossEncoder(args.SENTENCE_CROSSENCODER, max_length=512)
    print('Loading GNews')
    google_news = GNews()
    print('Loaded GNews, ready to get news')
    if args.mode == 'get_news': # warning: this will take a long time (multiple weeks)
        for nct_id in tqdm(studies['nct_id']):
            # print(f'Getting news for {nct_id}')
            time.sleep(random.random()*5+4)
            if os.path.exists(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json")):
                # print(f'{nct_id} already exists')
                with open(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json"), 'rb') as f:
                    news = json.load(f)
                if len(news) > 0:
                    print(f'Loaded {len(news)} news for {nct_id}')
                    continue

            news = google_news.get_news(f"{nct_id} clinical trial")
            # news = getNewsData(f"{nct_id} clinical trial")
            if news is None:
                news = []
            if len(news) > 0:
                print(f'Got {len(news)} news for {nct_id}')
            with open(os.path.join(args.SAVE_NEWS_LOG_PATH, nct_id+".json"), "w") as f:
                json.dump(news, f)


    # # ======================== Process the news data ========================
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
