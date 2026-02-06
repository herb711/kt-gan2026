#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run main experiment
cd /root/workcopy/kt-gan2026/

# Data Preprocessing Example
python data_preprocess.py --dataset_name assist2009 --min_seq_len 3 --maxlen 200 --kfold 5

python main.py --data_path "/root/workcopy/datasets/data/assist2009/train_valid_sequences.csv"
