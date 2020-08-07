from graphics import *
import helper as hp
from table import Table


class DefaultPage(Container):
    def __init__(self):
        view = Color(hp.backgroundColor)
        super().__init__(view=view)


class SimplePage(Container):
    def __init__(self):
        view = ZStack(views=[
            Rect(color=hp.blue, cornerRadius=10),
            Table(filePath="examples/small.csv", border=0)
        ])
        super().__init__(view=view)


class RunPage(Container):
    def __init__(self):

        self.tableView = Table(filePath="examples/animal.csv", fontSize=20)

        def updateTable(sender):
            self.tableView = self.tableView.replace(Table(filePath="examples/" + sender.view.text))

        view = VStack(views=[
            Label("Title"),
            HStack(views=[
                VStack(views=[
                    Button(view=Label(fileName), altView=Label("Clicked!!"), run=updateTable) for fileName in hp.getFiles("examples", ext="csv")
                ], border=0),
                VStack(views=[
                    HStack(views=[
                        self.tableView,
                        ZStack(views=[
                            Rect(color=hp.green, strokeColor=hp.red, strokeWidth=3),
                            Label("Model")
                        ], border=0)

                    ], ratios=[0.65, 0.35], border=0),
                    Color(hp.backgroundColor)
                ], ratios=[0.7, 0.3], border=0)
            ], ratios=[0.15, 0.85], border=0)
        ], ratios=[0.05, 0.95], border=0)
        super().__init__(view=view)
