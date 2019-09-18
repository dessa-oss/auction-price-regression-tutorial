import boto3
import numpy as np
import pandas as pd

def download_data():
    s3 = boto3.resource('s3')
    bucket_name = 'offer-optimization'
    s3.meta.client.download_file(bucket_name, 'offer_outcomes.csv', 'data/offer_outcomes.csv')
    offer_outcomes_df = pd.read_csv('data/offer_outcomes.csv')
    return offer_outcomes_df

def action_selection(model, inputs_df, n_actions):
    predicted_action_scores_list = list()
    for action in range(n_actions):
        inference_df = inputs_df.copy()
        inference_df['actions'] = action
        preds = model.predict(inference_df)
        predicted_action_scores_list.append(preds)
    predicted_action_scores = np.array(predicted_action_scores_list)
    best_actions = np.argmax(predicted_action_scores, axis=0)
    return best_actions.flatten()