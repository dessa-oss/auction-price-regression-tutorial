
#  Atlas - Auction Price Regression Tutorial 

Backlink: https://docs.atlas.dessa.com/en/latest/tutorials/auction_price_regression_tutorial/

*Estimated time: 30 minutes*

## Introduction

This tutorial demonstrates how to make use of the features of [Atlas](https://www.github.com/dessa-research/atlas). Note that **any machine learning
job can be run in Atlas without modification.** 

However, with minimal changes to the code we can take advantage of
Atlas features that will enable us to:

* view artifacts such as plots and tensorboard logs, alongside model performance metrics  
* launch many training jobs at once
* organize model experiments more systematically

This tutorial assumes that you have already installed Atlas. If you have not, then you can download Foundations
 Atlas community edition for free from [this link](https://www.atlas.dessa.com/).

## The Data

In this tutorial we make use of this data from a [Kaggle competition](https://www.kaggle.com/c/bluebook-for-bulldozers).
However don't worry about manually downloading the data as it will be downloaded automatically by running the provided
scripts. This dataset contains the sale price of heavy machinery (such as bulldozers) as well as it's usage, equipment 
type, and configuration. In this tutorial we'll train regression models to predict the sale price from the other
provided information. Note that the target (the sale price) has been mapped to a log scale.

## Start Atlas

Activate the conda environment in which Atlas is installed. Then run ```atlas-server start```. Validate that
 the GUI has been started by accessing it at http://localhost:5555/projects or replace `localhost` with IP of the rempte server you are running Atlas on.

## Clone the Tutorial

Clone this repository and make it your current directory by running:

```
git clone https://github.com/DeepLearnI/auction-price-regression-tutorial auction_price_regression_tutorial
cd auction_price_regression_tutorial
```

## Enabling Atlas Features

You are provided with the following python scripts:
* **driver.py**: A driver script which downloads the dataset, prepares it for model training and evaluation, trains a  
fully connected network with entity embeddings, then evaluates the model on the test set
* **model.py**: Code to implement the neural network

Note that this code runs without any modification.

To enable Atlas features, we only to need to make a few changes. Firstly add the
following line to the top of driver.py and model.py:

```
import foundations
```

### Logging Metrics and Parameters

The last line of driver.py prints the test mean squared error. We'll replace this print
statement with a call to the function `foundations.log_metric()`.This function takes two arguments, a key and a value. Once a
job successfully completes, **logged metrics for each job will be visible from the Foundations GUI.** Copy the following line
and replace the print statement with it.

Line 46 in driver.py:

```
foundations.log_metric('test mean squared error', float(mse))
```   

### Saving Artifacts

Currently, we create a matplotlib graph of validation mean squared error at the end of driver.py.  
With Atlas, we can save any artifact to the GUI with just one line. Add the following lines after `plt.savefig()`
to send the locally saved plot to the Atlas GUI.

Line 38 in driver.py:

```
foundations.save_artifact('plots/validation_mse.png', "validation_mse")
```   

### TensorBoard Integration

[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is a super powerful data visualization tool that makes visualizing your training extremely easy. Foundations
Atlas has full TensorBoard integration. To access TensorBoard directly from the Atlas GUI, add the following line of code
to start of driver.py.  

Line 8 in driver.py:

```
foundations.set_tensorboard_logdir('train_logs')
```

The function set_tensorboard_logdir() take one argument, the directory that your TensorBoard files will be saved to. TensorBoard files
are generated at each epoch through a callback, you can find the code in train() function model.py.

### Configuration

Lastly, create a file in the project directory named "job.config.yaml", and copy the text from below into the file.

```
project_name: 'bulldozer-demo'
log_level: INFO
```

## Running a Job

Activate the environment in which you have foundations installed, then from inside the project directory (bulldozer-demo)
run the following command:

```
foundations submit scheduler . driver.py
```

This will schedule a job to be run. Now open the Atlas GUI in your browser: http://localhost:5555/projects. Click into
the project 'bulldozer-demo', then click on the "Job Details" tab. Here, you'll see the running job. Once it completes, it will have a green status and you will
see your logged metrics.

<img src="images/gui.png" >

To view your saved artifacts, you can click on the expansion icon to the right of the running job, then click on the
"Artifacts" tab, and select the artifact you want to view from the menu below.

To view your model training on TensorBoard, simply select the running job, and click the "Send to TensorBoard" button on the GUI.

## Running a Hyperparameter Search

Atlas makes running and tracking the results of a set of hyperparameters easy. Create a new file called
'hyperparameter_search.py' and paste in the following code:

```
import os
os.environ['FOUNDATIONS_COMMAND_LINE'] = 'True'
import foundations
import numpy as np

def generate_params():
  hyperparameters = {'n_epochs': int(np.random.choice([2,4])),
                     'batch_size': int(np.random.choice([64,128])),
                     'validation_percentage': 0.1,
                     'dense_blocks': [{'size': int(np.random.choice([64,128,512])), 'dropout_rate': np.random.uniform(0,0.5)}],
                     'embedding_factor': np.random.uniform(0.2,0.6),
                     'learning_rate':np.random.choice([0.0001,0.001,0.0005]),
                     'lr_plateau_factor':0.1,
                     'lr_plateau_patience':3,
                     'early_stopping_min_delta':np.random.choice([0.0001,0.001]),
                     'early_stopping_patience':5}


  return hyperparameters

# A loop that calls the submit method from the Foundations SDK which takes the hyperparameters and the entrypoint script for our code (driver.py)

num_jobs = 5
for _ in range(num_jobs):
  hyperparameters = generate_params()
  foundations.submit(scheduler_config='scheduler', job_dir='.', command='driver.py', params=hyperparameters, stream_job_logs=True)
```

This script samples hyperparameters uniformly from pre-defined ranges, then submits jobs using those hyperparameters. For a script that exerts more control over the hyperparameter sampling, check the end of the tutorial.
The job execution code is still coming from driver.py; i.e. each experiment is submitted to and ran with the driver.

In order to get this to work, a small modification needs to be made to driver.py. In the code block where the hyperparameters are defined (indicated by the comment 'define
hyperparameters'), we'll load the sampled hyperparameters instead of defining a fixed set of hyperparameters explictely.

Replace that block (line 19 - 28) with the following:

```
# define hyperparameters
hyperparameters = foundations.load_parameters()
```

Now, to run the hyperparameter search, from the project directory (bulldozer-demo) simply run

```
python hyperparameter_search.py
```

## Congrats!

That's it! You've completed the Foundations Atlas Tutorial. Now, you should be able to go to the GUI and see your
running and completed jobs, compare model hyperparameters and performance, as well as view artifacts and training
visualizations on TensorBoard.

Do you have any thoughts or feedback for Foundations Atlas? Join the [Dessa Slack community](https://dessa-community.slack.com/join/shared_invite/enQtNzY5MTA3OTMxNTkwLWUyZDYzM2JmMDk0N2NjNjVhZDU5NTc1ODEzNzJjMzRlMDcyYmY3ODI1ZWMxYTQ3MzdmNjcyOTVhMzg2MjkwYmY)!

## License
```
Copyright 2015-2020 Square, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

© 2020 Square, Inc. ATLAS, DESSA, the Dessa Logo, and others are trademarks of Square, Inc. All third party names and trademarks are properties of their respective owners and are used for identification purposes only.