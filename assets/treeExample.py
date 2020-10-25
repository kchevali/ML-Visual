from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# TODO guide to install libraries above
#pip install sklearn
#pip install numpy
#pip install pandas

# get the file path from the user
fileName = input("Enter file path:")

# get the column header name from the user that will be targeted
targetCol = input("Enter target column:")

# get the number of columns to be used
colNum = int(input("Num of cols:"))

# ask the user for the column headers to use
columns = [targetCol] + [input("Col:") for _ in range(colNum)]

# create model object from sklearn
model = DecisionTreeClassifier()
# data = pd.get_dummies(pd.read_csv(fileName)[columns])

# read data from file
data = pd.read_csv(fileName)[columns]

# split the data 70% - training and 30% - testing
train, test = train_test_split(data, test_size=0.3)

# train model with training data (70%)
model.fit(train, train[targetCol])

# test model with testing data(30%)
acc = metrics.accuracy_score(test, model.predict(test))

# output final accuracy
print("Accuracy:", acc)
