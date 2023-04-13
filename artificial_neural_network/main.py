import numpy as np
import pandas as pd
import tensorflow as tf

DATASET_PATH = 'Churn_Modelling.csv'

## Read the dataset
dataset = pd.read_csv(DATASET_PATH)

## Filter only necessary features, remove columns like: ID, customer name...
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
