from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing, utils
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# TODO guide to install libraries above
# pip install sklearn
# pip install numpy
# pip install pandas

# get the file path from the user
fileName = input("Enter file path:")

# get x column header name
x_column = input("Enter X column name:")

# get y column header name
y_column = input("Enter Y column name:")

# create model object from sklearn
model = LogisticRegression()

# read data from file
data = pd.read_csv(fileName)[[x_column] + [y_column]]
# data = preprocessing.LabelEncoder().fit_transform(raw_data)

# split the data 70% - training and 30% - testing
train, test = train_test_split(data, test_size=0.3)

# train model with training data (70%)
train_x = train[x_column].to_numpy().reshape(-1, 1)
train_y = train[y_column].to_numpy().reshape(-1, 1)
model.fit(train_x, train_y)

# test model with testing data(30%)
test_x = test[x_column].to_numpy().reshape(-1, 1)
test_y = test[y_column].to_numpy().reshape(-1, 1)
err = metrics.mean_squared_error(test_x, model.predict(test_y))

# output final error
print("Error:", err)
