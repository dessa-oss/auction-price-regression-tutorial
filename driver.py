from model import FullyConnectedNetwork
from utils import download_data
import matplotlib.pyplot as plt

# add foundations.set_tensorboard_logdir() code here

# download data
train_df, test_df = download_data()

# prepare data
input_size = len(train_df.columns) - 1 # don't include the target when counting inputs
numeric_columns = ['machine_hours_current_meter', 'age_in_years', 'target']
categorical_sizes = {col: train_df[col].nunique() for col in train_df.columns if col not in numeric_columns}

# define hyperparameters
# replace following with foundations.load_parameters()
hyperparameters = {'n_epochs': 5,
                   'batch_size': 128,
                   'validation_percentage': 0.1,
                   'dense_blocks': [{'size': 256, 'dropout_rate': 0}],
                   'embedding_factor': 0.5,
                   'learning_rate':0.0001,
                   'lr_plateau_factor':0.1,
                   'lr_plateau_patience':3,
                   'early_stopping_min_delta':0.001,
                   'early_stopping_patience':5}

# train
model = FullyConnectedNetwork(input_size, hyperparameters, categorical_sizes)
hist = model.train(train_df)


val_mse_history = hist.history['val_mean_squared_error']
plt.plot(list(range(1, len(val_mse_history)+1)), val_mse_history)
plt.savefig('plots/validation_mse.png')

# add foundations.save_artifact() code here

# evaluate model performance on test set
mse = model.evaluate(test_df)
print("test mean squared error: ", mse)

# add foundations.log_metric() code here

