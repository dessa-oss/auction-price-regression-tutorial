import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Lambda, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, History, TensorBoard

class FullyConnectedNetwork:

    def __init__(self, input_size, hyperparameters, categorical_sizes):
        self.hyperparameters = hyperparameters
        self.categorical_sizes = categorical_sizes
        self.history = History()
        inputs = Input(shape=(input_size,))
        embedding_layers = list()
        for i, col_name in enumerate(sorted(list(categorical_sizes.keys()))):
            categorical_size = categorical_sizes[col_name]
            embedding_size = int(categorical_size ** (hyperparameters['embedding_factor']))
            ith_input_slice = Lambda(lambda x: x[:, i])(inputs)
            embedding = Embedding(categorical_size, embedding_size, input_length=1)(ith_input_slice)
            embedding_layers.append(embedding)
        numeric_inputs_slice = Lambda(lambda x: x[:, len(categorical_sizes):])(inputs)
        to_concat = embedding_layers + [numeric_inputs_slice]
        all_inputs = Concatenate(axis=1)(to_concat)
        hidden_input = all_inputs
        for block_params in self.hyperparameters['dense_blocks']:
            hidden_output = Dense(block_params['size'], activation='relu')(hidden_input)
            hidden_output = Dropout(block_params['dropout_rate'])(hidden_output)
            hidden_input = hidden_output
        outputs = Dense(1, activation='linear')(hidden_output)
        self.model = Model(inputs, outputs)
        # define optimization procedure
        self.lr_annealer = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=hyperparameters['lr_plateau_factor'],
                                             patience=hyperparameters['lr_plateau_patience'], verbose=1)
        self.early_stopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=hyperparameters['early_stopping_min_delta'],
                                           patience=hyperparameters['early_stopping_patience'], verbose=1)

        self.tensorboard = TensorBoard(log_dir='train_logs', histogram_freq=1)
        self.model.compile(optimizer=Adam(lr=hyperparameters['learning_rate']),
                           loss='mean_squared_error',
                           metrics=['mean_squared_error'])


    def preproc_train(self, train_df):
        train_inputs = train_df.drop('target', axis=1)
        all_cols_set = set(train_inputs.columns)
        categorical_cols_set = set(list(self.categorical_sizes.keys()))
        self.non_categorical_cols = list(all_cols_set - categorical_cols_set)
        self.column_order = sorted(list(categorical_cols_set)) + sorted(self.non_categorical_cols)
        # normalize non-categorical columns
        self.non_categorical_train_mean = train_inputs[self.non_categorical_cols].mean(axis=0)
        self.non_categorical_train_std = train_inputs[self.non_categorical_cols].std(axis=0)
        train_inputs[self.non_categorical_cols] -= self.non_categorical_train_mean
        train_inputs[self.non_categorical_cols] /= self.non_categorical_train_std
        # ensure that inputs are presented in the right order
        train_inputs = train_inputs[self.column_order]
        x_train = train_inputs.values
        y_train = train_df['target'].values
        # split training and validation
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,
                                                                        test_size=self.hyperparameters[
                                                                            'validation_percentage'])
        return x_train, y_train, x_validation, y_validation

    def train(self, train_df):
        x_train, y_train, x_validation, y_validation = self.preproc_train(train_df)
        return self.model.fit(x_train, y_train, epochs=self.hyperparameters['n_epochs'],
                       batch_size=self.hyperparameters['batch_size'],
                       validation_data=(x_validation, y_validation),
                       callbacks=[self.lr_annealer, self.early_stopper, self.history, self.tensorboard],
                       verbose=1)

    def preproc_inference(self, test_df):
        test_inputs = test_df.drop('target', axis=1)
        # normalize non-categorical columns
        test_inputs[self.non_categorical_cols] -= self.non_categorical_train_mean
        test_inputs[self.non_categorical_cols] /= self.non_categorical_train_std
        # ensure that inputs are presented in the right order
        test_inputs = test_inputs[self.column_order]
        x_test = test_inputs.values
        y_test = test_df['target'].values
        return x_test, y_test

    def predict(self, x_test):
        return self.model.predict(x_test).flatten()

    def evaluate(self, test_df):
        x_test, y_test = self.preproc_inference(test_df)
        preds = self.predict(x_test)
        return mean_squared_error(y_test, preds)
