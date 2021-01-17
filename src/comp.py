from enum import Enum
import numpy as np
from math import sqrt
from table import Table


class Data:
    def __init__(self, params=[], labelValues=None):  # , trainCount, testCount
        """
        params = [dict]
        """
        self.params = params
        self.labelValues = labelValues

        self.x = []
        self.y = []
        self.labels = []

        def func(x):
            return 0

        for param in self.params:
            pType = param["type"] if "type" in param else "single"
            count = param["count"] if "count" in param else 250

            if pType == "single":
                correlation = param["correlation"] if "correlation" in param else 0
                func1 = param["func"] if "func" in param else func
                self.store(*self.getPtsDouble(p1=param["x"], p2=param["y"], count=count, correlation=correlation), count=count, func=func1)
            else:
                correlationX = param["correlationX"] if "correlationX" in param else 0
                correlationY = param["correlationY"] if "correlationY" in param else 0
                func1 = param["func1"] if "func1" in param else func
                func2 = param["func2"] if "func2" in param else func

                x1, x2 = self.getPtsDouble(p1=param["x1"], p2=param["x2"], count=count, correlation=correlationX)
                y1, y2 = self.getPtsDouble(p1=param["y1"], p2=param["y2"], count=count, correlation=correlationY)
                self.store(x=x1, y=y1, count=count, func=func1)
                self.store(x=x2, y=y2, count=count, func=func2)

    def readDict(self, p):
        dist = p["dist"]
        if dist == "uniform":
            return (np.random.uniform, (p["min"], p["max"]))
        elif dist == "normal":
            return (np.random.normal, (p["mean"], p["std"]))

        # df = degrees of freedom
        return (np.random.standard_t, (p["df"]))

    def getPtsSingle(self, p, count):
        run, args = self.readDict(p)
        return run(*args, size=count)

    def getPtsCorr(self, p1, p2, count, correlation):
        gen1, (mean1, std1) = self.readDict(p1)
        gen2, (mean2, std2) = self.readDict(p2)
        pts1 = gen1(size=count)
        pts2 = gen2(size=count)
        return (mean1 + std1 * pts1, mean2 + std2 * (pts1 * correlation + pts2 * sqrt(1 - correlation * correlation)))  # x1,x3

    def getPtsDouble(self, p1, p2, count, correlation):
        if correlation != 0:
            return self.getPtsCorr(p1=p1, p2=p2, count=count, correlation=correlation)
        else:
            return self.getPtsSingle(p=p1, count=count), self.getPtsSingle(p=p2, count=count)

    def store(self, x, y, count, func):
        y += func(x)
        self.x.append(x)
        self.y.append(y)

        if self.labelValues != None:
            index = len(self.labels)
            while index >= len(self.labelValues):
                self.labelValues.append(self.labelValues[-1] + 1)
            self.labels.append(np.full(count, self.labelValues[index], dtype=np.int64))

    def getTable(self):
        x = np.concatenate(self.x)
        y = np.concatenate(self.y)

        if self.labelValues == None:
            p = {"target": "y", "columns": ["x"]}
            arr = [x, y]
        else:
            p = {"target": "label", "columns": ["x", "y"]}
            arr = [np.concatenate(self.labels), x, y]

        return Table(numpy=np.array(arr).transpose(), param=p)


if __name__ == '__main__':
    print("RUNNING COMP")

    def x2(x):
        return x * x

    def negx2(x):
        return -x * x

    dataOptions1 = [{
        "x": {
            "dist": "normal",
            "mean": 0,
            "std": 0.1
        },
        "y": {
            "dist": "normal",
            "mean": 0,
            "std": 0.1
        }
    }]
    dataOptions2 = [{
        "x": {
            "dist": "uniform",
            "min": 0,
            "max": 1
        },
        "y": {
            "dist": "normal",
            "mean": 0,
            "std": 1
        },
    }]
    dataOptions3 = [{
        "x": {
            "dist": "normal",
            "mean": 100,
            "std": 10
        },
        "y": {
            "dist": "normal",
            "mean": 0,
            "std": 400
        },
        "func": x2
    }]
    dataOptions4 = [{
        "x": {
            "dist": "normal",
            "mean": 0,
            "std": 1
        },
        "y": {
            "dist": "normal",
            "mean": 0,
            "std": 1
        },
        "correlation": 0.9
    }]

    dataOptions5 = [{
        "type": "double",
        "x1": {
                "dist": "normal",
                "mean": 5,
                "std": 2
        },
        "y1": {
            "dist": "normal",
            "mean": 0,
            "std": 8
        },
        "x2": {
            "dist": "normal",
            "mean": 3,
            "std": 1
        },
        "y2": {
            "dist": "normal",
            "mean": 50,
            "std": 10
        },
        "func1": x2
    }]

    training = Data(params=dataOptions5).getTable()
    # training, testing = table.partition()

    print("TRAINING")
    print(training.data)

    # print("TESTING")
    # print(testing.data)

    # import matplotlib.pyplot as plt
    # plt.scatter(training['x'], training['y'], c=training['label'], alpha=0.5)
    # plt.show()
