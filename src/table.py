import helper as hp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from graphics import Color


class Table:

    def __init__(self, df=None, param=None, filePath=None, numpy=None, features=None, constrainX=None, constrainY=None, drawTable=True, classColors=None):
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
        self.drawTable = drawTable

        # get column names
        self.yName = self.param['target']
        self.xNames = self.param['columns']
        if features != None:
            self.xNames = self.xNames[:features]
        self.features = len(self.xNames)
        self.colNames = [self.yName] + self.xNames
        self.constrainX = constrainX
        self.constrainY = constrainY

        # process data frame
        self.data = df if df is not None else (pd.read_csv(hp.resourcePath(self.filePath + ".csv")) if self.filePath != None else pd.DataFrame(data=numpy, columns=self.colNames))
        try:
            # this will only succeed if the data is hashable
            self.data = self.data.drop_duplicates(self.colNames)
        except:
            pass
        self.data = self.data[self.colNames]
        # print("Table Length:", len(self.data.index))

        # convert float labels to int
        # self.data[self.yName] = self.data[self.yName].astype('int64')

        # constraining is used for logistic model to map range min,max to 0,1
        if(self.constrainX != None):
            x = self.data[self.xNames]
            self.data[self.xNames] = self.constrainX[1] * (x - x.min()) / x.max() + self.constrainX[0]

        if(self.constrainY != None):
            y = self.data[self.yName]
            self.data[self.yName] = self.constrainY[1] * (y - y.min()) / y.max() + self.constrainY[0]

        self.dataX = self.data[self.xNames]
        self.dataY = self.data[self.yName]

        self.y = self.dataY.to_numpy()
        self.y = self.y.reshape([self.y.shape[0], 1])
        self.x = self.dataX.to_numpy()

        try:
            # this will only succeed if the data is hashable
            self.classSet = self.dataY.unique()
            self.classCount = len(self.classSet)
            self.isBoolCol = [self.data[colName].isin([0, 1]).all() for colName in self.colNames]

            if classColors != None:
                self.classColors = classColors
            else:
                self.classColors = {}
                i = 0
                for label in sorted(list(self.classSet)):
                    # using str(label) because Table.flatten() is used for Grid.items and it may convert to string
                    self.classColors[str(label)] = hp.calmColor(i / self.classCount)
                    i += 1
        except:
            self.classSet = None
            self.classCount = 0
            self.isBoolCol = []
            self.classColors = {}

        self.rowCount = len(self.data.index)  # total rows of data
        self.colCount = len(self.xNames)  # does not include target

        try:
            self.minX1 = self.data[self.xNames[0]].min()
            self.maxX1 = self.data[self.xNames[0]].max()
            x2Name = self.yName if self.features == 1 else self.xNames[1]
            self.minX2 = self.data[x2Name].min()
            self.maxX2 = self.data[x2Name].max()
        except:
            pass

        # self.selectedRow = None
        self.selectedCol = None  # index
        self.lockedCols = []
        self.graphics = []

        self.mapper = {d['column']: d['values'] for d in (self.param['map'] if "map" in self.param else {})}

        # Hide Code

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

    def init(self, df, param=None, filePath=None, features=None, constrainX=None,
             constrainY=None, drawTable=None, classColors=None):
        """
        Create a new table based on the given parameters or the current table
        """
        return Table(df=df, param=self.param if param == None else param, filePath=self.filePath if filePath == None else filePath, features=self.features if features == None else features, constrainX=self.constrainX if constrainX == None else constrainX,
                     constrainY=self.constrainY if constrainY == None else constrainY, drawTable=self.drawTable if drawTable == None else drawTable, classColors=self.classColors if classColors == None else classColors)

    def partition(self, testing=0.3) -> tuple:
        """
        Split the table into two tables by rows
        """
        train, test = train_test_split(self.data, test_size=testing)
        return self.init(train), self.init(test)

    def matchValue(self, colIndex, value):
        """
        Create a new table where the row contains a value at the given column.
        """
        return self.init(df=self[self[self.xNames[colIndex]] == value])

    def minX(self, column=None):
        """
        Get the minimum value in the x column.
        """
        return self.dataX.min()[column if column != None else self.xNames[0]]

    def maxX(self, column=None):
        """
        Get the maximum value in the x column.
        """
        return self.dataX.max()[column if column != None else self.xNames[0]]

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

    def tableChange(self, colIndex=None, isSelect=None, isLock=None, isNewTable=False, reset=False):
        """
        colIndex: index (x values starting at 1)
        isSelect: True, False or None(no change)
        isLock:True, False or None(no change)
        """

        if reset:
            self.selectedCol = None
            self.lockedCols = []
            colIndex = None
            isSelect = False
            isLock = False
            isNewTable = True
        else:

            # can't lock/select a none colIndex or unlock nothing
            # you can select what you unlock though
            if ((isLock or (isLock != False and isSelect)) and colIndex == None) or (isLock == False and len(self.lockedCols) == 0) or (isSelect == False and self.selectedCol == None):
                # print("Fail Change | Cond 1:", ((isLock or isSelect) and column == None), "Cond 2:", (isLock == False and len(self.lockedCols) == 0), "Cond 3:", (isSelect == False and self.selectedCol == None))
                return

            if isLock:
                self.lockedCols.append(colIndex)
            elif isLock == False:
                # you can select this item
                colIndex = self.lockedCols.pop()

            if isSelect != None:
                self.selectedCol = colIndex if isSelect else None

        # Change None value to previous values
        # isSelect = self.selectedCol != None and self.selectedCol == colIndex
        # isLock = colIndex in self.lockedCols
        for graphic in self.graphics:
            graphic.tableChange(colIndex=colIndex, isSelect=isSelect, isLock=isLock, isNewTable=isNewTable, reset=reset)

    def getEncodedData(self):
        """
        Creates a new Table where categorized data is converted to boolean by adding a column for each unique value per column.
        Based on pd.get_dummies()
        """
        dataX = pd.get_dummies(self.dataX)

        param = self.param.copy()
        param['columns'] = list(dataX.columns)
        return self.init(df=pd.concat([dataX, self.dataY], axis=1, join="inner"), param=param)

    def map(self, colIndex, value):
        column = self.colNames[colIndex]
        return self.mapper[column][value] if column in self.mapper and value in self.mapper[column] else str(value)

    def getPt(self, x, y, color):  # get many points
        return (hp.map(x, self.minX1, self.maxX1, -1, 1, clamp=False), hp.map(y, self.minX2, self.maxX2, -1, 1, clamp=False), color)

    def getPts(self):
        if self.drawTable:
            if self.features >= 2:
                return [self.getPt(self.x[i][0], self.x[i][1], self.classColors[str(self.y[i][0])]) for i in range(self.rowCount)]
            return [self.getPt(self.x[i][0], self.y[i], Color.green) for i in range(self.rowCount)]

    def __getitem__(self, column):
        return self.data[column]


class Table2D:
    """
    This object will store a 2D numpy array (x) and a 1D numpy array (y)
    """

    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)

    def init(self, x, y):
        self.x = x
        self.y = y
        self.shuffle()

    def partition(self, testing=0.3):
        length = int(len(self.x) * testing)
        return Table2D(self.x[:length], self.y[:length]), Table2D(self.x[length:], self.y[length:])

    def shuffle(self):
        shuffler = np.random.permutation(len(self.x))
        self.x = self.x[shuffler]
        self.y = self.y[shuffler]

    def digitsDataset():
        from sklearn.datasets import load_digits
        digits = load_digits()
        return Table2D(digits.data, digits.target)


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
