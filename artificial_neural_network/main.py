import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

DATASET_PATH = 'Churn_Modelling.csv'

## Read the dataset
dataset = pd.read_csv(DATASET_PATH)

## Filter only necessary features, remove columns like: ID, customer name...
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

## Encode categorical data (category to number)
# 'Gender' column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# 'Geography' column (One-hot encoding)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
