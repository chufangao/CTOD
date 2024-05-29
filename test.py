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
import datetime
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

# sys.path.append('./GNews/')
# from gnews import GNews