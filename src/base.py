
class TableBase:
    def __init__(self, table=None, partition=None, **kwargs):
        if table != None:
            self.setTable(table, partition=partition)

    def setTable(self, table, partition=None):
        if partition != None:
            self.table, self.testingTable = table.partition(testing=partition)
        else:
            self.table = table
            self.testingTable = None


class SingleModel(TableBase):
    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        if model != None:
            self.setModel(model)

    def setModel(self, model):
        self.model = model


class MultiModel(TableBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = []
        self.compModels = []

    def addModel(self, model):
        if model != None:
            self.models.append(model)

    def addCompModel(self, model):
        if model != None:
            self.compModels.append(model)
