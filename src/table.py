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
    def __init__(self, data=None, headers=None, filePath=None, params=None, xIndex=0, yIndex=1, canDisplay=True, **kwargs):
        if(data is not None):
            self.readData(data, headers)
        elif(filePath != None):
            self.readFile(filePath, params)
        else:
            raise Exception("Invalid Table args")

        self.canDisplay = canDisplay
        self.xIndex = xIndex
        self.yIndex = yIndex
        self.process()

    def readData(self, data, headers):
        self.data = data
        self.getSize()
        self.headers = headers if headers is not None else np.empty([self.cols], dtype=str)

    def readFile(self, filePath, params):
        file = np.genfromtxt(filePath + ".csv", dtype=float, delimiter=',', names=True)
        self.headers = np.array(file.dtype.names)

        self.params = hp.loadJSON(filePath + ".json") if not params else params
        if self.params != None:

            self.getSize()  # need number of rows to swap
            # swap cols based on param
            self.swapCols([np.nonzero(self.headers == header)[0][0] for header in self.params['columns']])

        # update column size
        self.getSize()

    def process(self):
        # move label to column 0
        np.random.shuffle(self.data)
        self.getMaxMin([self.xIndex, self.yIndex])
        self.minX, self.minY = tuple(self.minArr)
        self.maxX, self.maxY = tuple(self.maxArr)

    def getSize(self):
        self.rows, self.cols = self.data.shape[0], len(self.data[0])

    def getMaxMin(self, indicies):
        self.minArr, self.maxArr = [self.data[0][index] for index in indicies], [self.data[0][index] for index in indicies]
        for i in range(1, self.rows):
            for j in range(len(indicies)):
                index = indicies[j]
                self.minArr[j] = min(self.minArr[j], self.data[i][index])
                self.maxArr[j] = max(self.maxArr[j], self.data[i][index])

    def getUniqueColumn(self, index):
        col = set()
        for row in self.data:
            col.add(row[index])
        return col

    def getArray(self, indicies):
        arr = np.empty([self.rows, len(indicies)], dtype=float)
        for i in range(self.rows):
            for j in range(len(indicies)):
                arr[i][j] = self.data[i][indicies[j]]
                arr[i][j] = self.data[i][indicies[j]]
        # print(arr)
        return arr

    def createDisplayDots(self):
        pass

    def createView(self, createCell, **kwargs):
        return Grid(createView=createCell, cols=self.cols, rows=self.rows, **kwargs)

    def createDisplayX(self, x):
        return hp.map(x, self.minX, self.maxX, -1, 1)

    def createDisplayY(self, y):
        return hp.map(y, self.minY, self.maxY, -1, 1)

    def partition(self, testSize=0.5):
        testLength = int(self.rows * testSize)
        return Table(data=self.data[testLength:], headers=self.headers, xIndex=self.xIndex, yIndex=self.yIndex), Table(data=self.data[:testLength], headers=self.headers, xIndex=self.xIndex, yIndex=self.yIndex)

    def swapCols(self, indicies):
        self.headers = self._swap(self.headers, indicies)
        for i in range(self.rows):
            self.data[i] = tuple(self._swap(self.data[i], indicies))

    def _swap(self, row, indicies):
        return [row[indicies[i]] for i in range(len(indicies))]

    def getColumn(self, index):
        return [row[index] for row in self.data]

    def __iter__(self):
        return self.data.__iter__()

    def __str__(self):
        out = self.headers.__str__() + "\n"
        for row in self.data:
            out += row.__str__() + "\n"
        return out

    def __getitem__(self, item):
        return self.data[item]


class LabelledTable(Table):

    # input should be (headers + data) or (cols + rows)
    def __init__(self, **kwargs):
        super().__init__(xIndex=1, yIndex=2, **kwargs)
        self.classSet = self.getUniqueColumn(0)
        self.classCount = len(self.classSet)
        self.createDisplayDots()

    def createDisplayDots(self):
        self.classDots = {}
        for label in self.classSet:
            self.classDots[label] = []

        for label, x, y, *_ in self.data:
            self.classDots[label].append((self.createDisplayX(x), self.createDisplayY(y)))


class XYTable(Table):

    # input should be (headers + data) or (cols + rows)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.createDisplayDots()

    def createDisplayDots(self):
        self.dots = [(self.createDisplayX(x), self.createDisplayY(y)) for x, y in self.data]


if __name__ == '__main__':
    hp.clear()
    table = LabelledTable(filePath="examples/decisionTree/small")
    # a, b = table.partition(0.75)
    # table.swapCols(0, 1)
    print(table)
