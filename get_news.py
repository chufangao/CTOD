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
from sentence_transformers import SentenceTransformer

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

    # mode = 'get_news' # 'get_news' or 'process_news'
    mode = 'process_news'

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
        with open('./filtered_ticker_dict_642.pkl', 'rb') as f:
        # with open('./ticker_dict_211.pkl', 'rb') as f:
            ticker_dict = pickle.load(f)
        ticker_dict = {k.lower(): v for k, v in ticker_dict.items()}

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
            all_company_dfs.append(df)
        all_company_dfs = pd.concat(all_company_dfs)

        print("Processing sentiment")
        pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone", device='cuda')
        batch_size = 256

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
        encoder = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                                      cache_folder='/srv/local/data/chufan2/huggingface/')
        encoded_titles = encoder.encode(titles, convert_to_numpy=True, device='cuda')
        np.save('./news_title_embeddings.npy', encoded_titles)

        all_company_dfs.to_csv('./stock_news_logs/news.csv', index=False)