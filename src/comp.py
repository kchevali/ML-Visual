from enum import Enum
import numpy as np
from math import sqrt
from table import Table


class Feature:

    def __init__(self, mean=None, std=None, minRange=None, maxRange=None):
        """
        uniform: minRange/maxRange
        other: mean/std
        """
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
    def __init__(self, xDist, yDist, xFeatures, yFeatures, p=0.5, funcs=None):  # , trainCount, testCount
        self.xDist = xDist
        self.yDist = yDist
        self.xFeatures = xFeatures
        self.yFeatures = yFeatures
        self.p = p  # correlation coefficient
        self.genX = self.getGen(self.xDist)
        self.genY = self.getGen(self.yDist)
        self.funcs = [self.func for _ in range(len(xFeatures))] if funcs == None else funcs

    def generate(self, count):
        x_arr = self.genX(features=self.xFeatures, count=count)
        y_arr = self.genY(features=self.yFeatures, count=count)
        x = np.concatenate(x_arr)
        y = np.concatenate([y_arr[i] + self.funcs[i](x_arr[i]) for i in range(len(x_arr))])

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
        # np.array([labels, x, y])
        return Table(numpy=np.array(out).transpose(), param=param)

    def getGen(self, dist):
        if dist == Dist.Uniform:
            return self.uniform
        return self.normal if dist == Dist.Normal else self.tDist

    def uniform(self, features, count):
        return [np.random.uniform(features[i].a, features[i].b, count) for i in range(len(features))]

    def correlated(self, pts1, pts2, count, fx1, fx2):
        pts3 = pts1 * self.p + pts2 * sqrt(1 - self.p * self.p)  # x1 & x2 -> x3
        return [fx1.a + fx1.b * pts1, fx2.a + fx2.b * pts3]  # x1,x3

    def normal(self, features, count):
        return self.correlated(
            pts1=np.random.normal(0, 1, count),
            pts2=np.random.normal(0, 1, count),
            count=count,
            fx1=features[0],
            fx2=features[1]
        )

    def tDist(self, features, count):
        return self.correlated(
            pts1=np.random.standard_t(1, count),
            pts2=np.random.standard_t(1, count),
            count=count,
            fx1=features[0],
            fx2=features[1]
        )

    def func(self, x):
        return x


if __name__ == '__main__':
    print("RUNNING COMP")

    def x2(x):
        return x * x

    def negx2(x):
        return -x * x
    # print(np.full(10, 100))
    data = Data(xDist=Dist.Uniform, yDist=Dist.Normal, xFeatures=[
        Feature(minRange=0, maxRange=100),
        Feature(minRange=-50, maxRange=50)
    ], yFeatures=[
        Feature(mean=0, std=1000),
        Feature(mean=-50, std=50)
    ], p=1, funcs=[x2, negx2])

    # data = Data(xDist=Dist.Normal, yDist=Dist.T, xFeatures=[
    #     Feature(mean=0, std=0.5),
    #     Feature(mean=2, std=0.5)
    # ], yFeatures=[
    #     Feature(mean=0, std=0.5),
    #     Feature(mean=2, std=0.5)
    # ], p=0.25)

    training, testing = data.generate(1000).partition()

    print("TRAINING")
    print(training.data)

    # print("TESTING")
    # print(testing.data)

    import matplotlib.pyplot as plt
    plt.scatter(training['x'], training['y'], c=training['label'], alpha=0.5)
    plt.show()
