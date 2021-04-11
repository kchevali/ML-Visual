import numpy as np
from graphics import Color
import helper as hp
from math import sqrt, pi, sin, cos


class Model:
    def __init__(self, table, testingTable=None, name="", color=Color.red, critPtCount=0, isUserSet=False, isClassification=False, isRegression=False, **kwargs):
        self.name = name
        self.color = color
        self.critPtCount = critPtCount
        self.graphics = []  # stores Point objects
        self.graphicsDict = {}
        self.isUserSet = isUserSet
        self.isRunning = False
        self.isClassification = isClassification
        self.isRegression = isRegression
        self.setTable(table=table, testingTable=testingTable, isReset=False)

    def setTable(self, table, testingTable=None, isReset=True):
        self.table = table
        self.testingTable = testingTable
        # self.minX1, self.maxX1 = self.table.minX1, self.table.maxX1
        # self.minX2, self.maxX2 = self.table.minX2, self.table.maxX2
        if isReset:
            self.reset()

    def getPts(self, start=None, end=None, count=40):  # get many points
        return self.getSweepingPts(start=start, end=end, count=count)

    def addPt(self, x, y, storePt=True):
        if len(self.critPts) == 0 and not storePt:
            return
        if len(self.critPts) < self.critPtCount:
            self.critPts.append((x, y))
            self.getEq()
            self.updateGraphics()

            if not storePt:
                self.critPts.pop()
        elif storePt:
            self.reset()

    def getLinearPts(self, isLinear=True, stripeCount=False, m=None, b=None, **kwargs):
        edgePts = [
            (self.minX1, self.getY(self.minX1, **kwargs)),
            (self.maxX1, self.getY(self.maxX1, **kwargs)),
            (self.getX(self.minX2, **kwargs), self.minX2),
            (self.getX(self.maxX2, **kwargs), self.maxX2),
        ]
        if isLinear:
            if not stripeCount:
                return [self.table.getPt(x, y, self.color) for x, y in edgePts if(x >= self.minX1 and x <= self.maxX1 and y >= self.minX2 and y <= self.maxX2)]
            edgePts = sorted([(x, y) for x, y in edgePts if(x >= self.minX1 and x <= self.maxX1 and y >= self.minX2 and y <= self.maxX2)])
            if(len(edgePts) < 2):
                return []
            x, y = edgePts[0]
            endX, endY = edgePts[1]

            stripeLength = (endX - x) / (2 * stripeCount)
            dx = max(hp.quad(
                1 + m * m,
                -2 * (x + m * (y - b)),
                x * x + (y - b) * (y - b) - stripeLength * stripeLength
            )) - x

            alt = 1
            pts = [self.table.getPt(x, y, self.color)]

            x += dx
            y = self.getY(x, **kwargs)
            while x < endX:
                pts.append(self.table.getPt(x, y, self.color) if alt < 3 else None)
                x += dx
                y = self.getY(x, **kwargs)
                alt = (alt + 1) % 4
            if alt < 3:
                pts.append(self.table.getPt(endX, endY, self.color) if alt else None)
            return pts

        edges = []
        for x, y in edgePts:
            if type(x) != tuple:
                x = [x]
            if type(y) != tuple:
                y = [y]
            edges += [(x_, y_) for x_ in x for y_ in y if x_ != None and y_ != None and x_ >= self.minX1 and x_ <= self.maxX1 and y_ >= self.minX2 and y_ <= self.maxX2]
        edgePts = edges
        edgePts.sort()

        out = []
        for i in range(1, len(edgePts), 2):
            # sweep
            out += self.getSweepingPts(start=edgePts[i - 1][0], end=edgePts[i][0], count=20)
        return out

    def getSweepingPts(self, start=None, end=None, count=40):
        if start == None:
            start = self.minX1
        if end == None:
            end = self.maxX1
        return [self.table.getPt(num, self.getY(num), self.color) for num in hp.rangx(start, end, (end - start) / count, outputEnd=True)]

    def addGraphics(self, *args):
        for graphic in args:
            if type(graphic) == tuple:
                key, graphic = graphic
                self.graphicsDict[key] = graphic
            self.graphics.append(graphic)

    def getGraphic(self, key):
        return self.graphicsDict[key] if key in self.graphicsDict else None

    def startTraining(self):
        self.reset()
        self.isRunning = True

    def getScoreString(self):
        raise NotImplementedError("Please Implement getScoreString")

    def defaultScoreString(self):
        raise NotImplementedError("Please Implement defaultScoreString")

    def reset(self):
        self.critPts = []

    def getY(self, x):
        raise NotImplementedError("Please Implement getY")

    def getX(self, y):
        raise NotImplementedError("Please Implement getX")

    def getEq(self):
        pass

    def updateGraphics(self):
        pass


class Classifier(Model):
    # takes multiple features and outputs a categorical data
    def __init__(self, **kwargs):
        super().__init__(isLinear=False, isConnected=False, isClassification=True, **kwargs)
        # if len(self.table.xNames) > 0:
        #     self.colNameA = self.table.xNames[0]
        #     if len(self.table.xNames) > 1:
        #         self.colNameB = self.table.xNames[1]

    def accuracy(self, testTable):
        if testTable == None:
            raise Exception("Model cannot find accuracy if testTable is None")
        return self.run_accuracy(testTable.x, testTable.y)

    def run_accuracy(self, x, y):
        correct = 0
        count = len(x)
        for i in range(count):
            correct += 1 if self.predict(x[i]) == y[i] else 0
        return correct / count

    def predict(self, _x):
        pass

    def getScoreString(self):
        return self.name + " Acc: " + str(round(100 * self.accuracy(testTable=self.testingTable), 2)) + "%"

    def defaultScoreString(self):
        return self.name + " Acc: --"

    def getCircleLabelPts(self, table=None):  # circle
        if table == None:
            table = self.getTable()

        rowCount = table.rowCount
        if rowCount > 0:
            trig = 2.0 * pi / rowCount

        pts = []
        i = 0
        for label in table.y:
            pts.append((0.5 * cos(trig * i) if rowCount > 1 else 0.0,
                        0.5 * sin(trig * i) if rowCount > 1 else 0.0,
                        table.classColors[str(label[0])]))
            i += 1

        # if treeNode.parent != None:
        #     items.append(Label(text="{}:{}".format(treeNode.parent.column, treeNode.value), fontSize=20, color=Color.white, dx=-0.95, dy=-1))
        # return ZStack(items=items, keywords="dotStack", limit=150)
        return pts


class Regression(Model):
    # takes multiple features and outputs real data

    def __init__(self, **kwargs):
        super().__init__(isConnected=True, isRegression=True, **kwargs)
        self.colNameA, self.colNameB = self.table.xNames[0], self.table.yName
        self.reset()

    def error(self, testTable):
        error = 0
        for i in range(testTable.rowCount):
            error += (testTable.y[i] - self.getY(testTable.x[i]))**2
        return sqrt(error) / testTable.rowCount

    def reset(self):
        super().reset()
        self.cef = [0] * self.critPtCount  # highest power first

        pts = self.getGraphic("pts")
        if pts != None:
            pts.reset()

    def getEqString(self):
        raise NotImplementedError("Please Implement getEqString")

    def getScoreString(self):
        return self.name + " Error: " + str(round(self.error(testTable=self.testingTable), 4))

    def defaultScoreString(self):
        return self.name + " Error: --"

    def cefString(self, constant, power, showPlus=True, roundValue=2):
        while type(constant) == np.ndarray:
            constant = constant[0]
        constant = round(constant, roundValue)
        if constant == 0:
            return "0"
        return ("+" if constant > 0 and showPlus else "") + str(constant) + ("" if power <= 0 else ("x" + ("" if power == 1 else hp.superscript(power))))

    def updateGraphics(self):
        super().updateGraphics()
        self.getGraphic("pts").setPts(self.getPts())
        self.getGraphic("eq").setFont(text=self.getEqString())
        self.getGraphic("err").setFont(text="Error: " + self.getScoreString())


class SVMBase(Classifier):
    def __init__(self, **kwargs):
        super().__init__(critPtCount=3, **kwargs)

    def updateGraphics(self):
        super().updateGraphics()
        accGraphic = self.getGraphic("acc")
        if accGraphic != None:
            accGraphic.setFont(text=self.getScoreString())
            for i, pts in enumerate(self.getPts()):
                self.getGraphic("pts" + ("" if i == 0 else str(i + 1))).setPts(pts)

    def reset(self):
        super().reset()
        self.w = np.zeros([self.table.x.shape[1], 1])
        self.b = 0

    def getY(self, x, v):
        # returns a x2 value on line when given x1
        return (-self.w[0] * x - self.b + v) / self.w[1] if self.w[1] != 0 else 0

    def getX(self, y, v):
        # returns a x1 value on line when given x2
        return (-self.w[1] * y - self.b + v) / self.w[0] if self.w[0] != 0 else 0

    def getPts(self, start=None, end=None, count=40):  # get many points
        return [self.getLinearPts(
            isLinear=True,
            stripeCount=False if v == 0 else 30,
            m=-self.w[0] / self.w[1],
            b=(v - self.b) / self.w[1],
            v=v
        ) for v in [0, -1, 1]]
