import os
os.environ["HF_HOME"] = "/srv/local/data/chufan2/huggingface/"
import sys
from tqdm.auto import tqdm, trange
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import numpy as np
import pickle
import json
# import datetime
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append('./GNews/')
from gnews import GNews

#convert to datetime
def convert_to_datetime(date_str):
# Mon, 20 May 1996 07:00:00 GMT
    try:
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
    except:
        return pd.NA
    
# weeks = 13*20
def get_date_at_month(start_date, month_to_add):
    start_month_ = start_date[1] + month_to_add
    years = start_month_ // 12
    month = start_month_ % 12
    if month == 0:
        month = 12
        years -= 1
    return (start_date[0] + years, month, start_date[2])

def get_related_news(keyword, start_date, log_dir, num_months=240):
    google_news = GNews()
    all_results = {}
    for i in range(num_months):
        start_time = get_date_at_month(start_date, i)
        end_time = get_date_at_month(start_date, i+1)
        google_news.start_date = start_time
        google_news.end_date = end_time
        if datetime(*start_time) > datetime.now():
            print(f'Start date {start_time} is in the future')
            break
        # print(f'Getting news for {keyword} from {start_time} to {end_time}')
        # print(google_news.start_date, google_news.end_date)
        # Get the news results
        results = google_news.get_news(keyword)
        # random sleep to avoid getting blocked
        time.sleep(random.randint(1, 5)) # time.sleep(1)
        lens = len(results)
        print(f'Got {lens} news for {keyword} in {google_news.start_date} to {google_news.end_date}')
        all_results[str((start_time, end_time))] = results
        with open(log_dir+'news.json', "w") as f:
            json.dump(all_results, f)
        # dump the results

    # sorted_results= sorted(results, key=lambda x: datetime.strptime(x['published date'], "%a, %d %b %Y %H:%M:%S %Z"), reverse=True)

    return all_results

if __name__ == '__main__':
    # google_news = GNews()
    # pakistan_news = google_news.get_news('Pakistan')
    # print(pakistan_news[0])

    # mode = 'get_news' 
    # mode = 'process_news'
    mode = 'correspond_news_and_studies'
    print(f'Mode: {mode}')

    if mode == 'get_news':
        # stock_price_df = pd.read_csv("./stock_prices_635_industries.csv").drop(columns=['Unnamed: 0'])
        # stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
        # ticker_dict = pickle.load(open("./ticker_dict_211.pkl", "rb"))
        # filtered_ticker_dict = pickle.load(open("./filtered_ticker_dict_642.pkl", "rb"))
        # keys = list(filtered_ticker_dict.keys())
        # for f in keys:
        #     if (f in ticker_dict.keys()) or (filtered_ticker_dict[f] not in stock_price_df['Ticker'].unique()):
        #         del filtered_ticker_dict[f]
                
        # log_dir = './stock_news_logs/'
        # global_i = 0
        # for company, ticker in tqdm(filtered_ticker_dict.items()):

        #     min_date = stock_price_df[stock_price_df['Ticker'] == ticker]['Date'].min()
        #     start_date = (int(min_date.year), int(min_date.month), 1)

        #     os.makedirs(log_dir + ticker, exist_ok=True)
        #     news = get_related_news(company, start_date, num_months=12*50, log_dir=log_dir + ticker + '/')
        #     global_i += 1
        #     # break

        # get top 1000 most popular phase 3 industry sponsors
        data_path = './CITT/'
        sponsors = pd.read_csv(data_path + 'sponsors.txt', sep='|')
        studies = pd.read_csv(data_path + 'studies.txt', sep='|', low_memory=False)

        studies['study_first_submitted_date'] = pd.to_datetime(studies['study_first_submitted_date'])
        sponsors = pd.merge(sponsors, studies[['nct_id', 'phase', 'study_first_submitted_date']], on='nct_id', how='left')
        sponsors = sponsors[sponsors['agency_class']=='INDUSTRY']
        sponsors.dropna(inplace=True)
        sponsors = sponsors[sponsors['phase'].str.contains('Phase 3')]

        top_sponsors = sponsors['name'].value_counts().head(1000)
        coverage_ = top_sponsors.sum() / sponsors['name'].value_counts().sum()
        combined = pd.merge(top_sponsors.reset_index(),
                            sponsors.groupby('name')['study_first_submitted_date'].min().reset_index(),
                            on='name', how='left')

        with open('./filtered_ticker_dict_642.pkl', 'rb') as f:
            ticker_dict = pickle.load(f)
        ticker_df = pd.DataFrame(ticker_dict.items(), columns=['name', 'ticker'])
        combined = pd.merge(combined, ticker_df, on='name', how='left')
        # keep only na
        combined = combined[combined['ticker'].isna()]
        combined.sort_values('name', inplace=True)
        print(combined.shape, combined['name'])

        log_dir = './stock_news_logs/_names/'
        global_i = 0
        for name in tqdm(sorted(combined['name'])):
            if name.lower() in os.listdir(log_dir):
                print(f'{name} already exists')
            else:
                print(f'Getting news for {name}')
                min_date = combined[combined['name']==name]['study_first_submitted_date'].min()
                # print(date.year, date.month, date.day)
                start_date = (int(min_date.year), int(min_date.month), 1)

                os.makedirs(log_dir+ name.lower(), exist_ok=True)            
                news = get_related_news(name, start_date, num_months=12*50, log_dir=log_dir + name.lower() + '/')
            global_i += 1
            # break

    # ======================== Process the news data ========================
    elif mode == 'process_news':
        print('Processing news data')
        studies_df = pd.read_csv('./CITT/studies.txt', sep='|', low_memory=False)        
        ticker_dict_df = pd.read_csv('stock_price/ticker_dict_642.csv')
        ticker_dict = {row['name'].lower(): row['ticker'] for _, row in ticker_dict_df.iterrows()}

        all_company_dfs = []
        # for company, ticker in ticker_dict.items():
        #     if not os.path.exists(os.path.join('./stock_news_logs/_names/', ticker, 'news.json')):     # if exists
        #         continue
        #     with open(os.path.join('./stock_news_logs/_names/', ticker, 'news.json'), 'rb') as f:
        #         news = json.load(f)
        for company in sorted(os.listdir('./stock_news_logs/_names/')):
            if not os.path.exists(os.path.join('./stock_news_logs/_names/', company, 'news.json')):     # if exists
                print(f'{company} does not have news')
                continue
            with open(os.path.join('./stock_news_logs/_names/', company, 'news.json'), 'rb') as f:
                news = json.load(f)
            # print(company, ticker, news)

            all_titles = []
            all_descriptions = []
            all_dates = []
            all_publishers = []
            for k in news.keys():
                if len(news[k]) > 0:
                    # print(k, news[k][0].keys())
                    for i in range(len(news[k])):
                        date = convert_to_datetime(news[k][i]['published date'])
                        # print(news[k][i]['published date'],date); quit()
                        all_dates.append(date)
                        all_titles.append(news[k][i]['title'])
                        all_descriptions.append(news[k][i]['description'])
                        all_publishers.append(news[k][i]['publisher']['title'])

            df = pd.DataFrame({'date': all_dates, 'title': all_titles, 'description': all_descriptions, 'publisher': all_publishers})
            if company in ticker_dict.keys():
                ticker = ticker_dict[company]
            else:
                ticker = pd.NA
            df['ticker'] = ticker
            df['company'] = company
            all_company_dfs.append(df)
        all_company_dfs = pd.concat(all_company_dfs)
        # all_company_dfs.to_csv('./stock_price/news_tmp.csv', index=False); quit() # update old news.csv with company names

        print("Processing sentiment")
        pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone", device='cuda')
        batch_size = 512

        all_label_preds = []
        all_label_probs = []
        titles = all_company_dfs['title'].tolist()
        for i in trange(0, len(titles), batch_size):
            out = pipe(titles[i:i+batch_size], batch_size=batch_size)
            all_label_preds += [o['label'] for o in out]
            all_label_probs += [o['score'] for o in out]
        all_company_dfs['sentiment'] = all_label_preds
        all_company_dfs['sentiment_prob'] = all_label_probs

        # process title embeddings using pubmedbert
        print("Processing embeddings")
        encoder = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', cache_folder='/srv/local/data/chufan2/huggingface/')
        encoded_titles = encoder.encode(titles, convert_to_numpy=True, device='cuda', batch_size=batch_size)
        np.save('./stock_price/news_title_embeddings.npy', encoded_titles)

        encoded_studies = encoder.encode(studies_df['brief_title'].tolist(), convert_to_numpy=True, device='cuda', batch_size=batch_size)
        np.save('./stock_price/studies_title_embeddings.npy', encoded_studies)

        all_company_dfs.to_csv('./stock_price/news.csv', index=False)

        # crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, cache_folder='/srv/local/data/chufan2/huggingface/')
        # scores = crossencoder.predict(
        #     [("Query", "Paragraph1"), ("Query", "Paragraph2"), ("Query", "Paragraph3")]
        # )
    elif mode == 'correspond_news_and_studies':
        crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)
        encoder = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', cache_folder='/srv/local/data/chufan2/huggingface/')

        news_df = pd.read_csv('./stock_price/news.csv')
        news_title_embedding = np.load('./stock_price/news_title_embeddings.npy')
        studies = pd.read_csv('./CITT/studies.txt', sep='|', low_memory=False)
        sponsors = pd.read_csv('./CITT/sponsors.txt', sep='|')
        top_sponsors = pd.read_csv('stock_price/top_sponsors.csv')
        interventions = pd.read_csv('./CITT/interventions.txt', sep='|')
        conditions = pd.read_csv('./CITT/conditions.txt', sep='|')

        sponsors = sponsors[sponsors['name'].isin(top_sponsors['name'])]
        sponsors['name'] = sponsors['name'].str.lower()

        studies = studies[studies['nct_id'].isin(sponsors['nct_id'])]
        studies = studies[studies['nct_id'].isin(interventions['nct_id'])]
        studies = studies[studies['nct_id'].isin(conditions['nct_id'])]
        news_df['date'] = pd.to_datetime(news_df['date'])
        studies['completion_date'] = pd.to_datetime(studies['completion_date'])
        # # studies['intervention_name'] = studies['nct_id'].map(interventions.set_index('nct_id')['name'])
        # studies_title_embedding = studies_title_embedding[studies.index.values]

        # map intervention and condition to study name2
        interventions['name'] = interventions['name'].astype(str)
        intervention_names = interventions.groupby('nct_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
        intervention_names.columns = ['nct_id', 'intervention_name']
        condition_names = conditions.groupby('nct_id')['name'].apply(lambda x: ' '.join(x)).reset_index()
        condition_names.columns = ['nct_id', 'condition_name']
        studies = pd.merge(studies, intervention_names, on='nct_id', how='left')
        studies = pd.merge(studies, condition_names, on='nct_id', how='left')
        studies['title2'] = studies['intervention_name'] + ' ' + studies['condition_name']

        studies_title2_embedding = encoder.encode(studies['title2'], convert_to_numpy=True, device='cuda', show_progress_bar=True)
        np.save('./stock_price/studies_title2_embeddings.npy', studies_title2_embedding)

        # print(news_df.shape, news_title_embedding.shape, studies.shape, studies_title2_embedding.shape)
        # # most relevant news for each study

        # for each study, filter out news that are not within 2 year of completion date
        top_k = 10
        topk_cols = [f'top_{i}' for i in range(top_k, 0, -1)] + [f'top_{i}_sim' for i in range(top_k, 0, -1)]
        studies[topk_cols] = pd.NA
        column_ind = studies.columns.get_loc(f'top_{top_k}')

        for i in trange(studies.shape[0]):
            if studies.iloc[i]['completion_date'] is pd.NaT:
                continue
            # get sponsors of the study
            sponsors_ = sponsors[sponsors['nct_id'] == studies.iloc[i]['nct_id']]['name'].tolist()
            news_df_ = news_df[news_df['company'].isin(sponsors_)]
            news_df_ = news_df_[
                np.abs((news_df_['date'] - studies.iloc[i]['completion_date']).dt.days) < 365*2
                ]
            # print(news_df_.shape)
            if news_df_.shape[0] == 0:
                continue
            news_title_embedding_ = news_title_embedding[news_df_.index]
            
            similarity = studies_title2_embedding[i] @ news_title_embedding_.T
            inds = np.argsort(similarity)[-top_k:]
            sims = crossencoder.predict([(studies.iloc[i]['title2'], news_df_.iloc[ind]['title']) for ind in inds], show_progress_bar=False)

            # original inds are indices in news_df_, we need to convert them to indices in news_df
            studies.iloc[i, column_ind:column_ind+len(inds)] = news_df_.iloc[inds].index
            studies.iloc[i, column_ind+top_k:column_ind+top_k+len(news_df_)] = sims

        studies.to_csv('./stock_price/studies_with_news.csv', index=False)
