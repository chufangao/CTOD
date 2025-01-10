import pickle
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from io import StringIO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--CTTI_PATH', type=str, default='../CTTI/')
    args = parser.parse_args()
        
    studies = pd.read_csv(os.path.join(args.CTTI_PATH, 'studies.txt'), sep='|')
    interventions = pd.read_csv(os.path.join(args.CTTI_PATH, 'interventions.txt'), sep='|')

    interventions = interventions[interventions['intervention_type'].str.lower().isin(['drug', 'biological'])]
    studies = studies[studies['nct_id'].isin(interventions['nct_id'])]

    studies = studies[~studies['overall_status'].str.lower().isin(['terminated', 'withdrawn', 'suspended', 'withheld', 'no longer available', 'temporarily not available'])]

    studies = studies.dropna(subset=['phase'])

    chrome_options = Options()
    chrome_options.add_argument("--headless") #FOR DEBUG COMMENT OUT SO YOU CAN SEE WHAT YOU'RE DOING
    driver = webdriver.Firefox(options=chrome_options)

    amendment_counts = []
    for i, nct in enumerate(tqdm(studies['nct_id'])):
        try:
            driver.get(f'https://clinicaltrials.gov/study/{nct}?tab=history')
            # driver.page_source # needs to be called before the next line
            time.sleep(1)
            card_content = driver.find_element("class name","card-content").get_attribute('innerHTML')

            versions_df = pd.read_html(StringIO(card_content))[0]
            latest_version = versions_df['Version'].iloc[-2]

            amendment_counts.append([nct, latest_version])

            if i % 100 == 0:
                out_df = pd.DataFrame(amendment_counts, columns=['nct_id', 'amendment_count'])
                out_df.to_csv('./amendment_counts.csv', index=False)
        except Exception as e:
            print(f"Error for {nct}: {e}")
        # break    
        # Optional: Introduce a delay between requests
        #time.sleep(1)
