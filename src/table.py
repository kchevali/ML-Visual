import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Table:

    # input should be (headers + data) or (cols + rows)
    def __init__(self, df=None, param=None, filePath=None, numpy=None, features=None, constrainX=None, constrainY=None):
        """
        Create a Table to store data.

        Args 1:
        ----------
        - filePath: String - path to csv & param file
        - param: Dict? - overwrites param file

        Args 2:
        ----------
        - df: pandas.DataFrame - references dataframe w/o copy
        - filePath: String? - path to param file only
        - param: Dict? - overwrites param file

        Args 3:
        ----------
        - numpy: numpy.ndarray - generate DataFrame from numpy
        - filePath: String? - path to param file only
        - param: Dict? - overwrites param file

        Optional Args
        - features: int - number of x columns to keep. None=all
        - constrainX: (minX, maxX) - constrain all x cols to range
        - constrainY: (minY, maxY) - constrain y col to range
        """
        # get data parameters
        self.filePath = filePath
        self.fileName = self.filePath.split("/")[-1].split(".")[0] if filePath != None else None
        self.param = hp.loadJSON(filePath + ".json") if not param else param
        self.features = features

        # get column names
        self.yName = self.param['target']
        self.xNames = self.param['columns']
        if self.features != None:
            self.xNames = self.xNames[:self.features]
        self.colNames = [self.yName] + self.xNames
        self.constrainX = constrainX
        self.constrainY = constrainY

        # process data frame
        self.data = df if df is not None else (pd.read_csv(filePath + ".csv") if filePath != None else pd.DataFrame(data=numpy.transpose(), columns=self.colNames))
        self.data = self.data.drop_duplicates(self.colNames)
        self.data = self.data[self.colNames]

        # constraining is used for logistic model to map range min,max to 0,1
        if(self.constrainX != None):
            x = self.data[self.xNames]
            self.data[self.xNames] = self.constrainX[1] * (x - x.min()) / x.max() + self.constrainX[0]

        if(self.constrainY != None):
            y = self.data[self.yName]
            self.data[self.yName] = self.constrainY[1] * (y - y.min()) / y.max() + self.constrainY[0]

        self.y = self.data[self.yName].to_numpy()
        self.x = self.data[self.xNames].to_numpy()

        self.classSet = self.data[self.yName].unique()
        self.classCount = len(self.classSet)
        self.rowCount = len(self.data.index)  # total rows of data
        self.colCount = len(self.xNames)  # does not include target

        self.classColors = {}
        i = 0
        for label in sorted(list(self.classSet)):
            self.classColors[label] = hp.calmColor(i / self.classCount)
            i += 1

        # self.selectedRow = None
        self.selectedCol = None  # index
        self.lockedCols = []
        self.graphics = []

        # Hide Code
        # self.mapper = self.param['map'] if "map" in self.param else {}
        # self.columns = self.data.columns
        # self.loc = self.data.loc
        # self.x = np.array(self.x)
        # from sklearn.preprocessing import StandardScaler
        # self.x = StandardScaler().fit_transform(self.x)
        # self.y = np.array(self.y).reshape([self.rowCount, 1])

    def majorityInColumn(self, column):
        return self.data[column].value_counts().idxmax()

    def majorityInTargetColumn(self):
        return self.majorityInColumn(self.yName)

    def partition(self, testing=0.3) -> tuple:
        train, test = train_test_split(self.data, test_size=testing)

        trainTable = Table(df=train, param=self.param, filePath=self.filePath, features=self.features, constrainX=self.constrainX, constrainY=self.constrainY)
        testTable = Table(df=test, param=self.param, filePath=self.filePath, features=self.features, constrainX=self.constrainX, constrainY=self.constrainY)
        return trainTable, testTable

    def matchValue(self, colIndex, value):
        return Table(df=self[self[self.xNames[colIndex]] == value], param=self.param, filePath=self.filePath, features=self.features, constrainX=self.constrainX, constrainY=self.constrainY)

    def minX(self, column=None):
        return self.data[self.xNames].min()[column if column != None else self.xNames[0]]

    def maxX(self, column=None):
        return self.data[self.xNames].max()[column if column != None else self.xNames[0]]

    def minY(self):
        return self.y.min()

    def maxY(self):
        return self.y.max()

    def uniqueVals(self, colIndex):
        return self[self.xNames[colIndex]].unique()

    def flattenValues(self):
        return self.data.values.flatten()

    def flatten(self):
        return np.concatenate([self.colNames, self.flattenValues()])

    def addGraphic(self, graphic):
        self.graphics.append(graphic)

    def tableChange(self, column, isSelect=None, isLock=None, isNewTable=False):
        """
        column: index (x values starting at 1)
        isSelect: True, False or None(no change)
        isLock:True, False or None(no change)
        """
        # can't lock/select a none column or unlock nothing
        # you can select what you unlock though
        if ((isLock or (isLock != False and isSelect)) and column == None) or (isLock == False and len(self.lockedCols) == 0) or (isSelect == False and self.selectedCol == None):
            print("Fail Change | Cond 1:", ((isLock or isSelect) and column == None), "Cond 2:", (isLock == False and len(self.lockedCols) == 0), "Cond 3:", (isSelect == False and self.selectedCol == None))
            return

        if isLock:
            self.lockedCols.append(column)
        elif isLock == False:
            # you can select this item
            column = self.lockedCols.pop()

        if isSelect != None:
            self.selectedCol = column if isSelect else None

        # Change None value to previous values
        # isSelect = self.selectedCol != None and self.selectedCol == column
        # isLock = column in self.lockedCols
        for graphic in self.graphics:
            graphic.tableChange(column, isSelect, isLock, isNewTable)

    def __getitem__(self, column):
        return self.data[column]

    # Hidden Methods
        # def map(self, column, value):
    #     return self.mapper[column][value]
    # def createXYTable(self, xIndex=0):
    #     param = self.param.copy()
    #     param['columns'] = [self.xNames[xIndex]]
    #     return Table(df=self.data, param=param)
    # def createXXYTable(self, x1=0, x2=1):
    #     param = self.param.copy()
    #     param['columns'] = [self.xNames[x1], self.xNames[x2]]
    #     return Table(df=self.data, param=param)


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
    table = Table(filePath="examples/logistic/sigmoid", constrainX=(0, 1), constrainY=(0, 1))
    print("Row Count:", table.rowCount)
    print("Target:", table.yName)
    print("Features:", table.xNames)
    print("Classes(" + str(table.classCount) + "):", table.classSet)
    print()

    colName = table.xNames[0]
    print("Majority in", colName, "is", table.majorityInColumn(colName))
    print("Majority in", table.yName, "is", table.majorityInTargetColumn())
    print(colName, "Min:", table.minX(colName), "Max:", table.maxX(colName))
    print(table.yName, "Min:", table.minY(), "Max:", table.maxY())
    print("Head:")
    print(table.data.head())
    print("Flatten:")
    print(table.flatten())

    # print(table.createXYTable().data)
