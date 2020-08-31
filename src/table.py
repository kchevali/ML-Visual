import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
from graphics import *
from gui import g
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Table:

    # input should be (headers + data) or (cols + rows)
    def __init__(self, data=None, param=None, encodedData=None, filePath=None, canDisplay=True):
        # assert (data != None and param != None) or filePath != None, "Cannot generate table"
        self.param = hp.loadJSON(filePath + ".json") if not param else param
        self.targetName = self.param['target']
        self.indexCol = self.param['index'] if ("index" in self.param) else ""
        self.mapArray = self.param['map'] if "map" in self.param else []

        self.colNames = [self.targetName] + self.param['columns']

        self.data = data if data is not None else pd.read_csv(filePath + ".csv")  # , index_col=self.indexCol
        self.data = self.data[self.colNames]  # move target to front and limit to given columns
        self.encodedData = encodedData if encodedData is not None else pd.get_dummies(self.data)
        try:
            self.encodeTargetCol = self.encodedData[self.targetName]
        except:
            # TODO - why crash for examples/movie (when click headerSelection)
            self.encodeTargetCol = self.data[self.targetName]
        # print(self.encodedData)
        # map values if obtained from file
        mappedTargetCol = False
        if filePath:
            for mapDict in self.mapArray:
                column = mapDict['column']
                if column == self.targetName:
                    mappedTargetCol = True

                columnDict = {int(k): v for k, v in mapDict['values'].items()}
                self.data[column] = self.data[column].map(columnDict)
            self.data = self.data.replace({
                True: 'Yes',
                False: 'No'
            })
        # if not mappedTargetCol:
        #     self.encodeTargetCol = self.data[self.targetName]

        self.targetCol = self.data[self.targetName]
        self.columns = self.data.columns
        self.loc = self.data.loc
        self.dataIndex = self.data.index
        self.classes = self.targetCol.unique()
        self.classCount = self.targetCol.nunique()

        self.dataRows = len(self.data)  # total rows of data

    def createView(self, createCell, **args):
        return Grid(createView=createCell, cols=len(self.columns), rows=len(self.data) + 1, **args)

    def majorityValue(self, column):
        return self.data[column].value_counts().idxmax()

    def commonTarget(self):
        return self.majorityValue(self.targetName)

    def partition(self, testing=0.3):
        train, test = train_test_split(self.data, test_size=testing)

        trainEncode = self.encodedData.loc[train.index]
        testEncode = self.encodedData.loc[test.index]

        trainTable = Table(data=train, encodedData=trainEncode, param=self.param)
        testTable = Table(data=test, encodedData=testEncode, param=self.param)
        return trainTable, testTable

    def iterrows(self):
        return self.data.iterrows()

    def __getitem__(self, key):
        return self.data[key]


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
