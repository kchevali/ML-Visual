from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

fileName = input("Enter file path:")
targetCol = input("Enter target column:")
colNum = int(input("Num of cols:"))
columns = [targetCol] + [input("Col:") for _ in range(colNum)]

model = DecisionTreeClassifier()
# data = pd.get_dummies(pd.read_csv(fileName)[columns])
data = pd.read_csv(fileName)[columns]
train, test = train_test_split(data, test_size=0.3)
model.fit(train, train[targetCol])
acc = metrics.accuracy_score(test, model.predict(test))
print("Accuracy:", acc)
