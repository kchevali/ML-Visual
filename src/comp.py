from enum import Enum
import numpy as np
from math import sqrt
from table import Table


class Feature:

    def __init__(self, mean=None, std=None, minRange=None, maxRange=None):
        self.a = mean if mean != None else minRange
        self.b = std if std != None else maxRange


# class Dataset:
#     def __init__(self, x, y, labels):
#         self.x = x
#         self.y = y
#         self.labels = labels

#     def zip(self):
#         return np.transpose(np.array([self.x, self.y, self.labels]))


class Dist(Enum):
    T = 0
    Normal = 1
    Uniform = 2


class Data:
    def __init__(self, dist, xFeatures, yFeatures, p=0.5):  # , trainCount, testCount
        self.dist = dist
        self.xFeatures = xFeatures
        self.yFeatures = yFeatures
        # self.trainCount = trainCount
        # self.testCount = testCount
        self.p = p  # correlation coefficient
        self.generate = self.selectGenMethod()
        # self.generatePoints()
        # self.points = [TRAIN,TEST] : TRAIN-[(x,y)-several classes]

    def selectGenMethod(self):
        if(self.dist == Dist.Uniform):
            return self.uniform
        if(self.dist == Dist.Normal):
            return self.normal
        return self.tDist

    # def generatePoints(self):
    #     self.training = self.generate(self.trainCount)
    #     self.testing = self.generate(self.testCount)

    def uniform(self, count):
        x, y = [], []
        for i in range(len(self.xFeatures)):
            fx = self.xFeatures[i]
            fy = self.yFeatures[i]
            x.append(np.random.uniform(fx.a, fx.b, count))
            y.append(np.random.uniform(fy.a, fy.b, count))
        return self.createTable(np.concatenate(x), np.concatenate(y), count)

    def correlated_single(self, pts1, pts2, count, fx1, fx2):
        pts3 = pts1 * self.p + pts2 * sqrt(1 - self.p * self.p)  # x2
        return np.concatenate([fx1.a + fx1.b * pts1, fx2.a + fx2.b * pts3])

    def correlated(self, ptsX, ptsY, count):
        assert len(self.xFeatures) == 2 and len(self.yFeatures) == 2, "Can't correlate when the number of features != 2"
        x = self.correlated_single(*ptsX, count, self.xFeatures[0], self.xFeatures[1])
        y = self.correlated_single(*ptsY, count, self.yFeatures[0], self.yFeatures[1])
        return self.createTable(x, y, count)

    def normal(self, count):
        return self.correlated(
            (np.random.normal(0, 1, count), np.random.normal(0, 1, count)),
            (np.random.normal(0, 1, count), np.random.normal(0, 1, count)),
            count)

    def tDist(self, count):
        return self.correlated(
            (np.random.standard_t(1, count), np.random.standard_t(1, count)),
            (np.random.standard_t(1, count), np.random.standard_t(1, count)),
            count)

    def createTable(self, x, y, count):
        labels = []
        for i in range(len(self.xFeatures)):
            labels.append(np.full(count, i, dtype=np.int64))
        labels = np.concatenate(labels)

        # out = [[labels[i], x[i], y[i]] for i in range(len(x))]
        out = [labels, x, y]

        param = {
            "target": "label",
            "columns": [
                "x",
                "y"
            ]
        }
        #np.array([labels, x, y])
        return Table(numpy=np.array(out).transpose(), param=param)


if __name__ == '__main__':

    # print(np.full(10, 100))
    data = Data(dist=Dist.Normal, xFeatures=[
        Feature(minRange=0, maxRange=100),
        Feature(minRange=-50, maxRange=50)
    ], yFeatures=[
        Feature(minRange=0, maxRange=100),
        Feature(minRange=-50, maxRange=50)
    ])

    data = Data(Dist.T, xFeatures=[
        Feature(mean=0, std=0.5),
        Feature(mean=2, std=0.5)
    ], yFeatures=[
        Feature(mean=0, std=0.5),
        Feature(mean=2, std=0.5)
    ], p=0.25)

    training, testing = data.generate(100).partition()

    print("TRAINING")
    print(training.data)

    # print("TESTING")
    # print(testing.data)

    # import matplotlib.pyplot as plt
    # plt.scatter(training['x'], training['y'], c=training['label'], alpha=0.5)
    # plt.show()
