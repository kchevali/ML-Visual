from models import Classifier, Regression
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class LibModel:

    def __init__(self, lib, **kwargs):
        self.lib = lib


class LibDT(Classifier, LibModel):

    def __init__(self, **kwargs):
        LibModel.__init__(self, lib=DecisionTreeClassifier())
        Classifier.__init__(self, **kwargs)
        self.fit()

    def fit(self):
        x = self.table.dataX
        y = self.table.dataY
        self.lib.fit(x, y)

    def predict(self, x):
        return self.lib.predict(x.reshape(1, -1))

    def accuracy(self, testTable):
        if testTable == None:
            raise Exception("Model cannot find accuracy if testTable is None")
        x = testTable.x
        y = testTable.y
        return self.run_accuracy(x, y)


class LibSVM(Classifier, LibModel):

    def __init__(self, **kwargs):
        # LibModel.__init__(self, lib=make_pipeline(StandardScaler(), SVC(gamma='auto')))
        LibModel.__init__(self, lib=SVC(kernel='linear', C=1000))
        Classifier.__init__(self, **kwargs)
        # self.fit()

    def fit(self):
        x = self.table.dataX
        y = self.table.dataY
        self.lib.fit(x, y)

        self.w = self.lib.coef_[0]
        self.b = self.lib.intercept_[0]

        self.isRunning = False

        accGraphic = self.getGraphic("acc")
        if accGraphic != None:
            accGraphic.setFont(text=self.getScoreString())
            for i, pts in enumerate(self.getPts()):
                self.getGraphic("pts" + ("" if i == 0 else str(i + 1))).setPts(pts)

    def predict(self, x):
        return self.lib.predict(x.reshape(1, -1))

    def accuracy(self, testTable):
        if testTable == None:
            raise Exception("Model cannot find accuracy if testTable is None")
        x = self.table.x
        y = self.table.y
        return self.run_accuracy(x, y)

    def getY(self, x, v):
            # returns a x2 value on line when given x1
        return ((-self.w[0] * x - self.b + v) / self.w[1])

    def getX(self, y, v):
        # returns a x1 value on line when given x2
        return (-self.w[1] * y - self.b + v) / self.w[0]

    def getPts(self, start=None, end=None, count=40):  # get many points
        return [self.getLinearPts(
            isLinear=True,
            stripeCount=False if v == 0 else 30,
            m=-self.w[0] / self.w[1],
            b=(v - self.b) / self.w[1],
            v=v
        ) for v in [0, -1, 1]]


if __name__ == '__main__':
    print("Running LIB MODELS")

    from table import Table
    import pandas as pd
    table = Table(filePath="examples/svm/svm_iris")
    table, testing = table.partition(testing=0.3)

    model = LibDT(table=table, testingTable=testing)
    print(model.getScoreString())

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_blobs

    # we create 40 separable points
    X, y = table.x, table.y

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)

    print("SV:", clf.support_vectors_)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins // draws lines
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors // draw circles
    # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
