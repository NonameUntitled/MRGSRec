# This is sourse code for the submission 567 "MRGSRec: Enhancing recommender systems performance with leveraging of multiple data representations”
# Instructions to run experiments

## Preparing datasets
Datasets used for experiments:
- Movielens 1M (https://grouplens.org/datasets/movielens):
- Amazon (https://nijianmo.github.io/amazon/index.html):
  - All_Beauty, Clothing_Shoes_and_Jewelry, Sports_and_Outdoors
 To initiate datasets preparation process, one needs to run notebooks from the [notebooks](./notebooks.zip) arhchive.

## Model training
To train the MRGSRec model, simply run:
```shell
python train.py --params /path/to/the/json/file/with/config
```
The script has 1 input argument: `params` which is the path to the json file with model configuration. The example of such file can be found [here](./configs/mrgsrec_train_config.json). This json file contains a nested dictionary with model hyperparameters and data preparation instructions. It should contain the following keys:

-`experiment_name` Name of the experiment

-`dataset` Information about the dataset

-`dataloader` Settings for dataloader

-`model` Model hyperparameters

-`optimizer` Optimizer hyperparameters

-`loss` Naming of different loss components
