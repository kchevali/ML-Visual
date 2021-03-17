from base_models import Classifier, Regression, SVMBase
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


class LibSVM(SVMBase, LibModel):

    def __init__(self, kernel='linear', **kwargs):
        # LibModel.__init__(self, lib=make_pipeline(StandardScaler(), SVC(gamma='auto')))
        self.kernel = kernel
        LibModel.__init__(self, lib=SVC(kernel=kernel, C=1000))
        SVMBase.__init__(self, **kwargs)
        # self.fit()

    def fit(self):
        x = self.table.dataX
        y = self.table.dataY
        self.lib.fit(x, y)
        self.isRunning = False

        if self.kernel == 'linear':
            self.w = self.lib.coef_[0]
            self.b = self.lib.intercept_[0]
            self.updateGraphics()

    def predict(self, x):
        return self.lib.predict(x.reshape(1, -1))


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
