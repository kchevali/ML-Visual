from base_models import Classifier, Regression, SVMBase
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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


class LibANN(Classifier, LibModel):
    def __init__(self, **kwargs):
        LibModel.__init__(self, lib=MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                                                  solver='sgd', tol=1e-4, random_state=1,
                                                  learning_rate_init=.1, verbose=True))
        Classifier.__init__(self, **kwargs)
        self.fit()

    def fit(self):
        self.lib.fit(self.table.x, self.table.y)

    def predict(self, x):
        return self.lib.predict(x.reshape(1, -1))


if __name__ == '__main__':
    print("Running LIB MODELS")

    from table import Table2D
    table = Table2D.digitsDataset()
    train, test = table.partition()
    ann = LibANN(table=train, testingTable=test)
    print("Accuracy:", ann.getScoreString())
