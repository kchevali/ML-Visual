from graphics import *
import helper as hp
from table import Table
from decisionTree import DecisionTree


class DefaultPage(Container):
    def __init__(self):
        view = Color(Color.backgroundColor)
        super().__init__(view=view)


class TestPage(Container):
    def __init__(self):
        view = ZStack(views=[
            Rect(color=Color.green, dx=-1, lockedWidth=500, lockedHeight=500),
            Rect(color=Color.yellow, dx=-1, lockedWidth=125, lockedHeight=125),
            Rect(color=Color.red, dx=-1, lockedWidth=50, lockedHeight=50),
            Rect(color=Color.green, dx=-1, lockedWidth=20, lockedHeight=20)
        ])
        super().__init__(view=view)


class SimplePage(Container):
    def __init__(self):
        view = VStack(views=[
            None,
            Rect(color=Color.red, lockedWidth=20, lockedHeight=300),
            Rect(color=Color.red, lockedWidth=20, lockedHeight=300)
        ], ratios=[0.5, 0.25, 0.25])
        # view = HStack(views=[
        #     None,
        #     VStack(views=[
        #         None,
        #         Rect(color=Color.red, lockedWidth=20, lockedHeight=400),
        #         Rect(color=Color.red, lockedWidth=20, lockedHeight=180)
        #     ]),
        #     VStack(views=[
        #         Rect(color=Color.blue, lockedWidth=20, lockedHeight=400),
        #         Rect(color=Color.blue, lockedWidth=400, lockedHeight=20),
        #         None
        #     ]),
        # ])
        print(view)
        super().__init__(view=view)


class RunPage(Container):
    def __init__(self):

        # Init Button Methods
        def updateTable(sender):
            self.setTable(Table(filePath="examples/" + sender.name, fontSize=20))
            self.updateContainers()

        def move(sender):
            sender.tag()
            self.table = self.tree.getTable()
            self.table.setContainer(self.tableContainer)
            self.tableContainer.updateAll()
            self.treeContainer.updateAll()
            self.updateMoveButtons()
            self.updateHeaderSelectionButtons()

        # Declare Containers
        self.tableContainer = None
        self.treeContainer = None
        self.colSelContainer = None

        self.treeMove = None

        # Set Table
        self.setTable(Table(filePath="examples/movie.csv", fontSize=20))
        self.treeMove = HStack(views=[
            ZStack(views=[
                Rect(color=Color.steelBlue),
                Button(view=Label("Back", keywords="backLabel"), tag=self.tree.goBack, run=move, keywords="backButton"),
            ]),
            ZStack(views=[
                Rect(color=Color.steelBlue),
                Button(view=Label("Top", keywords="topLabel"), tag=self.tree.goLeft, run=move, keywords="topButton"),
            ]),
            ZStack(views=[
                Rect(color=Color.steelBlue),
                Button(view=Label("Bottom", keywords="bottomLabel"), tag=self.tree.goRight, run=move, keywords="bottomButton")
            ])
        ])
        self.updateMoveButtons()

        # Build View
        view = VStack(views=[
            ZStack(views=[
                Rect(color=Color.steelBlue),
                Label("Decision Tree")
            ]),
            HStack(views=[
                VStack(views=[
                    ZStack(views=[
                        Rect(color=Color.steelBlue, cornerRadius=10),
                        Button(view=Label(fileName.split(".")[0]), altView=Label("Clicked!!"), name=fileName, run=updateTable)
                    ]) for fileName in hp.getFiles("examples", ext="csv")
                ]),
                VStack(views=[
                    HStack(views=[
                        self.table,
                        self.tree.getView()

                    ], ratios=[0.65, 0.35]),
                    VStack(views=[
                        None,
                        self.headerSelection,
                        self.treeMove
                    ])

                ], ratios=[0.7, 0.3])
            ], ratios=[0.15, 0.85])
        ], ratios=[0.08, 0.92])
        # print(view)
        super().__init__(view=view)

        # Define Containers
        self.tableContainer = self.table.container
        self.treeContainer = self.tree.getContainer()
        self.colSelContainer = self.headerSelection.container

    def setTable(self, table):
        self.table = table
        self.table.setContainer(self.tableContainer)
        self.tree = DecisionTree(table)
        self.tree.getView().setContainer(self.treeContainer)

        def selectColumn(sender):
            self.table.selectColumn(value=sender.isOn, index=sender.tag)
            rect = sender.getCousin(0)
            if sender.isOn:
                self.tree.add(self.tree.current.dataFrame.columns.values[sender.tag])
                rect.strokeColor = Color.white
            else:
                self.tree.remove()
                rect.strokeColor = Color.gray
            self.updateMoveButtons()
            self.treeContainer.updateAll()
            self.updateHeaderSelectionButtons()
            # print("SELECT:", self.tree.getView().container)

        columns = self.table.data.columns
        self.headerSelection = HStack(views=[
            ZStack(views=[
                Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
                Button(view=Label(columns[i], fontSize=15, color=Color.white),
                       toggleView=Label(columns[i], fontSize=15, color=Color.black),
                       #    altView=Label("{} CLK".format(columns[i]), fontSize=15),
                       isOn=False, tag=i, run=selectColumn)
            ]) for i in range(1, len(columns))

        ])
        self.headerSelection.setContainer(self.colSelContainer)
        self.updateMoveButtons()
        self.updateHeaderSelectionButtons()

    def updateMoveButtons(self):
        if self.treeMove:
            self.treeMove[0].isHidden = self.tree.isRoot()
            self.treeMove[1].isHidden = not self.tree.hasLeft()
            self.treeMove[2].isHidden = not self.tree.hasRight()

            self.treeMove[0].isDisabled = self.tree.isRoot()
            self.treeMove[1].isDisabled = not self.tree.hasLeft()
            self.treeMove[2].isDisabled = not self.tree.hasRight()

            if self.tree.isVertical():
                self.treeMove.keyDown("topLabel").setFont("Top")
                self.treeMove.keyDown("bottomLabel").setFont("Bottom")
            else:
                self.treeMove.keyDown("topLabel").setFont("Left")
                self.treeMove.keyDown("bottomLabel").setFont("Right")
            if self.treeMove.container:
                self.treeMove.container.updateAll()

    def updateHeaderSelectionButtons(self):
        columns = self.table.data.columns
        for c in self.headerSelection.containers:
            rect = c.view.getView(0)
            button = c.view.getView(1)
            column = columns[button.tag]

            if self.tree.isParentColumn(column):
                rect.strokeColor = Color.gray
                button.isDisabled = True
            else:
                button.isDisabled = False
                if self.tree.isCurrentColumn(column):
                    rect.strokeColor = Color.yellow
                else:
                    rect.strokeColor = Color.darkGray
                    button.setOn(isOn=False)
                    self.table.selectColumn(value=False, index=button.tag)

    def updateContainers(self):
        self.tableContainer.updateAll()
        self.colSelContainer.updateAll()
        self.treeContainer.updateAll()
