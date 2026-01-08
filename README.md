# DEPRECATED, PLEASE SEE https://github.com/sunlabuiuc/CTO

<p align="center"><img src="./CTO.png"/></p>

# CTO

Code for Automatically Labeling Clinical Trial Outcomes: A Large-Scale Benchmark for Drug Development

[Website: chufangao.github.io/CTOD](https://chufangao.github.io/CTOD/)

[Paper Link (Arxiv): Automatically Labeling Clinical Trial Outcomes: A Large-Scale Benchmark for Drug Development](https://arxiv.org/abs/2406.10292)

[![Dataset Link](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/chufangao/CTO)

Please see [Tutorials](https://github.com/chufangao/CTOD/tree/main/tutorials) if you want to play around with CTOD as soon as possible. **This includes off-the-shelf Google Collab notebooks!**
**

Modules for Weakly Supervised Labeling Functions
This repository provides implementations for various sources of weakly supervised labeling functions (LFs) used in the CTO benchmark. Below are the modules, each corresponding to a different data source and approach for generating weak labels:

- **[llm_prediction_on_pubmed](https://github.com/chufangao/CTOD/tree/main/llm_prediction_on_pubmed)** focuses on leveraging PubMed abstracts linked to clinical trials. We utilize abstracts from the Derived and Results categories, prioritizing the top two abstracts based on title similarity to the official trial title for relevance. We then employ the gpt-3.5 model with carefully designed prompts to predict clinical trial outcomes based on the abstract content. The specific prompts used for these tasks are detailed in the CTO paper.

- **[clinical_trial_linkage](https://github.com/chufangao/CTOD/tree/main/clinical_trial_linkage)** addresses the critical challenge of linking clinical trials across different phases (Phase 1, 2, 3) and to FDA approvals. Recognizing the inherent difficulties due to unstructured data, reporting inconsistencies, and variations in intervention details, we make a systematic effort to connect different phases of clinical trials and match Phase 3 trials with subsequent FDA approvals. E.g. if a similar trial is found in phase 3 from a trial in phase 2, then they are considered "linked". We use a reranking method after the initial text similarity retrieval to further refine relevance. 

- **[news_headlines](https://github.com/chufangao/CTOD/tree/main/news_headlines)** utilizes news headlines as a source of weak supervision. We performed extensive web scraping from Google News using SerpAPI, targeting headlines related to the top 1000 industry sponsors (covering approximately 80% of industry-sponsored trials). To extract sentiment, we employed FinBERT to classify the financial sentiment of these headlines as 'Positive' or 'Negative' (discarding 'Neutral'). For accurate news/trial matching, we implemented a retrieved top K filtering using text simlarity (PubMedBERT) to identify relevant headlines.

- **[stock_price](https://github.com/chufangao/CTOD/tree/main/stock_price)** explores stock price fluctuations as an indicator of market sentiment towards clinical trial outcomes. We hypothesize that stock prices of pharmaceutical and biotech companies often reflect market expectations.  We gather historical stock data from Yahoo Finance for companies associated with completed trials (where public tickers are available).  To mitigate short-term noise, we calculate a 5-day Simple Moving Average (SMA) of stock prices. We then compute the slope of this SMA over a 7-day window immediately following the trial completion date. This slope captures the direction and magnitude of stock price trends, providing a weak signal related to trial outcomes.

- **[labeling](https://github.com/chufangao/CTOD/tree/main/labeling)** describes our label generation process, which combines both unsupervised and supervised approaches. For each weakly supervised source of supervision, or labeling function (LF), derived from the modules above, we determine optimal quantile thresholds specific to Phases 1, 2, and 3 trials. These thresholds are fine-tuned on the TOP training dataset. Furthermore, the lfs.py script contains additional labeling functions derived from readily available clinical trial metrics. These metrics include trial status, number of adverse events reported, number of amendments made to the trial protocol, and more. Our final labeling strategy involves training a Random Forest model using the outputs of other weakly supervised LFs as features. 

## Reference
```bash
@article{gao2024automatically,
  title={Automatically Labeling Clinical Trial Outcomes: A Large-Scale Benchmark for Drug Development},
  author={Gao, Chufan and Pradeepkumar, Jathurshan and Das, Trisha and Thati, Shivashankar and Sun, Jimeng},
  journal={arXiv preprint arXiv:2406.10292},
  year={2024}
}
```    
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

[DOI: 10.57967/hf/4597](https://doi.org/10.57967/hf/4597)


### Authors

- [@chufangao](https://www.github.com/chufangao)
- [@Jathurshan0330](https://www.github.com/Jathurshan0330)
- [@trishad2](https://www.github.com/trishad2)

### Special Thanks

A huge thanks to [SerpApi](https://serpapi.com/) for their powerful search API--an invaluable resource for scalably gathering clinical trial news, making our research faster and more efficient. Specificially, we used SerpAPI to search more than 80,000 clinical trials using their Google search API.
