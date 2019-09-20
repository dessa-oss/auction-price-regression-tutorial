import os
os.environ['FOUNDATIONS_COMMAND_LINE'] = 'True'
import foundations
import numpy as np
import copy


class SearchSpace:

    def __init__(self, min, max, type):
        self.min = min
        self.max = max
        self.type = type

    def sample(self):
        if self.type == int:
            return np.random.randint(self.min, self.max)
        elif self.type == float:
            return round(np.random.uniform(self.min, self.max), 2)


def sample_hyperparameters(hyperparameter_ranges):
    hyperparameters = copy.deepcopy(hyperparameter_ranges)
    for hparam in hyperparameter_ranges:
        if isinstance(hyperparameter_ranges[hparam], SearchSpace):
            search_space = hyperparameter_ranges[hparam]
            hyperparameters[hparam] = search_space.sample()
        elif isinstance(hyperparameter_ranges[hparam], list):
            for i, block in enumerate(hyperparameter_ranges[hparam]):
                for block_hparam in block:
                    if isinstance(block[block_hparam], SearchSpace):
                        search_space = block[block_hparam]
                        hyperparameters[hparam][i][block_hparam] = search_space.sample()
    return hyperparameters


hyperparameter_ranges = {'n_epochs': 2,
                   'batch_size': 128,
                   'validation_percentage': 0.1,
                   'dense_blocks': [{'size': SearchSpace(64,512,int), 'dropout_rate': SearchSpace(0,0.5,float)}],
                   'embedding_factor': SearchSpace(0.2,0.6,float),
                   'learning_rate':0.0001,
                   'lr_plateau_factor':0.1,
                   'lr_plateau_patience':3,
                   'early_stopping_min_delta':0.001,
                   'early_stopping_patience':5}

num_jobs = 5
for _ in range(num_jobs):
    hyperparameters = sample_hyperparameters(hyperparameter_ranges)
    foundations.submit(scheduler_config='scheduler', job_dir='.', command='driver.py', params=hyperparameters, stream_job_logs=True)