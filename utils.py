import boto3
import numpy as np
import pandas as pd
import os

def download_data():
    s3 = boto3.resource('s3')
    bucket_name = 'bulldozer'
    if not os.path.isfile('data/train.csv'):
        s3.meta.client.download_file(bucket_name, 'train.csv', 'data/train.csv')
    if not os.path.isfile('data/test.csv'):
        s3.meta.client.download_file(bucket_name, 'test.csv', 'data/test.csv')
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    return train_df, test_df
