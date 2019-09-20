from model import FullyConnectedNetwork
from utils import download_data
from tensorflow.keras.callbacks import History

# download data
train_df, test_df = download_data()

# prepare data
input_size = len(train_df.columns) - 1 # don't include the target when counting inputs
numeric_columns = ['machine_hours_current_meter', 'age_in_years', 'target']
categorical_sizes = {col: train_df[col].nunique() for col in train_df.columns if col not in numeric_columns}

# train
hyperparameters = {'n_epochs': 5,
                   'batch_size': 256,
                   'validation_percentage': 0.1,
                   'dense_blocks': [{'size': 1, 'dropout_rate': 0.2}]}
model = FullyConnectedNetwork(input_size, hyperparameters, categorical_sizes)
model.train(train_df)

history = History()
print(history)

# evaluate model performance on test set
accuracy, roc_auc = model.evaluate(test_df)
print("test accuracy: ", accuracy)
print("test roc-auc: ", roc_auc)

