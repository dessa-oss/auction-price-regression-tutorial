from sklearn.model_selection import train_test_split
from model import FullyConnectedNetwork
from utils import download_data

offer_outcomes_df = download_data()
categorical_sizes = {'offer': offer_outcomes_df['offer'].nunique()}

# prepare data
offer_outcomes_df = offer_outcomes_df.rename({'outcome': 'target'}, axis=1)
input_size = len(offer_outcomes_df.columns) - 1 # don't include the target when counting inputs
train_df, test_df = train_test_split(offer_outcomes_df, test_size=0.15)

# train a model
hyperparameters = {'n_epochs': 1,
                   'batch_size': 32,
                   'validation_percentage': 0.1,
                   'dense_blocks': [{'size': 128, 'dropout_rate': 0.2}]}
model = FullyConnectedNetwork(input_size, hyperparameters, categorical_sizes)
model.train(train_df)

# evaluate model performance on test set
roc_auc = model.evaluate(test_df)
print(roc_auc)