import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Table:

    # input should be (headers + data) or (cols + rows)
    def __init__(self, df=None, param=None, filePath=None, numpy=None):
        # assert (data != None and param != None) or filePath != None, "Cannot generate table"
        self.param = hp.loadJSON(filePath + ".json") if not param else param
        self.yName = self.param['target']
        self.xNames = self.param['columns']
        self.mapper = self.param['map'] if "map" in self.param else {}
        self.colNames = [self.yName] + self.xNames
        self.data = df if df is not None else (pd.read_csv(filePath + ".csv") if filePath != None else pd.DataFrame(data=numpy.transpose(), columns=self.colNames))
        self.data = self.data.drop_duplicates(self.colNames)
        self.data = self.data[self.colNames]

        self.yData = self.data[self.yName]
        self.xData = self.data[self.xNames]

        self.columns = self.data.columns
        self.loc = self.data.loc
        self.classSet = self.yData.unique()
        self.classCount = self.yData.nunique()
        self.rowCount = len(self.data)  # total rows of data
        self.colCount = len(self.columns)

        self.xs = np.array(self.xData)
        from sklearn.preprocessing import StandardScaler
        self.xs = StandardScaler().fit_transform(self.xs)
        self.y = np.array(self.yData).reshape([self.rowCount, 1])

        self.classColors = {}
        i = 0
        for label in sorted(list(self.classSet)):
            self.classColors[label] = hp.calmColor(i / self.classCount)
            i += 1

    def majorityInColumn(self, column):
        return self.data[column].value_counts().idxmax()

    def majorityInTargetColumn(self):
        return self.majorityValue(self.yName)

    def partition(self, testing=0.3):
        train, test = train_test_split(self.data, test_size=testing)

        trainTable = Table(df=train, param=self.param)
        testTable = Table(df=test, param=self.param)
        return trainTable, testTable

    def map(self, column, value):
        return self.mapper[column][value]

    def minX(self, xIndex=0):
        return self.xData.min()[self.xNames[xIndex]]

    def maxX(self, xIndex=0):
        return self.xData.max()[self.xNames[xIndex]]

    def minY(self):
        return self.yData.min()

    def maxY(self):
        return self.yData.max()

    def createXYTable(self, xIndex=0):
        param = self.param.copy()
        param['columns'] = [self.xNames[xIndex]]
        return Table(df=self.data, param=param)

    def createXXYTable(self, x1=0, x2=1):
        param = self.param.copy()
        param['columns'] = [self.xNames[x1], self.xNames[x2]]
        return Table(df=self.data, param=param)

    def iterrows(self):
        return self.data.iterrows()

    def __getitem__(self, key):
        return self.data[key]


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
    table = Table(filePath="examples/decisionTree/animal")
    print(table.createXYTable().data)
