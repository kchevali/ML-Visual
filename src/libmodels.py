from models import Classifier, Regression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class LibDT(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lib = DecisionTreeClassifier()
        self.fit()

    def fit(self):
        # self.lib.fit(self.table.x, self.table.y)
        x = pd.get_dummies(self.table.data[self.table.xNames])
        y = pd.get_dummies(self.table.data[self.table.yName])
        self.lib.fit(x, y)

    def predict(self, x):
        return self.lib.predict(x.reshape(1, -1))

    def accuracy(self, testTable):
        if testTable == None:
            raise Exception("Model cannot find accuracy if testTable is None")
        x = pd.get_dummies(testTable.data[self.table.xNames]).to_numpy()
        y = pd.get_dummies(testTable.data[self.table.yName]).to_numpy().reshape(-1, 1)
        return self.run_accuracy(x, y)


if __name__ == '__main__':
    print("Running LIB MODELS")

    from table import Table
    import pandas as pd
    table = Table(filePath="examples/decisionTree/dt_movie")
    table, testing = table.partition(testing=0.4)

    model = LibDT(table=table, testingTable=testing)
    print(model.getScoreString())
