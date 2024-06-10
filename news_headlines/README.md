# News Headlines

## Prerequisites

- Download the trial dataset from [CTTI](https://aact.ctti-clinicaltrials.org/download) <path>. If it has already been downloaded, provide the path to the data in the scripts.
- We use [GNews](https://github.com/ranahaani/GNews) to scrap Google News for the news headlines.
- Please look at get_news.py and ensure that the paths to the CTTI downloads are specified, 

## 1. Scrape Google News

First, we extract trial features from the CITI dataset. Provide the <data_path> for downloaded CITI data in the command below:

```jsx
cd news_headlines
git clone https://github.com/ranahaani/GNews.git
```

Run the following command to start the scraping for the top 1000 industry sponsors (NOTE: This will take a long time, i.e. on the scale of multiple weeks.) We share our scraped headlines in the zenodo page.

```jsx
python get_news.py --mode=get_news
```

## 2. Obtaining Sentiment Embeddings from News Headlines and Study Titles

Running this command also saves the news title embeddings and a dataframe of the news as news.csv.
```jsx
python get_news.py --mode=process_news
```

## 3 Corresponding News and Trials: Encoding Study Title Emebddings and TopK Similarity

Running this command also saves study title embeddings.
 ```jsx
python get_news.py --mode=correspond_news_and_studies 
```