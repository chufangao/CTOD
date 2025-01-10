#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python biobert_trial_outcome.py --model=dmis-lab/biobert-base-cased-v1.2
CUDA_VISIBLE_DEVICES=2 python biobert_trial_outcome.py --model=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract