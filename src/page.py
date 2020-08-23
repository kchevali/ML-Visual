from graphics import *
import helper as hp
from table import Table
from decisionTree import DecisionTree


def createMouseDebug():
    return ZStack(items=[
        Rect(color=Color.white, keywords="mRect", border=0),
        Label(text="", fontSize=15, color=Color.black, keywords="text")
    ], lockedWidth=80, lockedHeight=20, dx=-1, dy=1)


class DefaultPage(ZStack):
    def __init__(self):
        views = [
            Color(Color.backgroundColor)
        ]
        super().__init__(items=views)


class SimplePage(ZStack):
    def __init__(self):
        def say(sender):
            print("Hello World!")

        views = [
            Rect(color=Color.blue)
        ]
        super().__init__(items=views)


class MainPage(ZStack):
    def __init__(self):

        firstPage = ExcerisePage()

        views = [
            HStack(items=[
                self.createTaskList(),
                firstPage
            ], ratios=[0.15, 0.85])
        ]
        self.pageContainer = firstPage.container
        # print(view)
        super().__init__(items=views)

    def createTaskList(self):
        tasks = ["Introduction", "Example", "Improvement", "Practice", "More Info"]
        pages = [IntroDTPage, ExcerisePage, IntroDTPage, ExcerisePage, IntroDTPage]

        def selectTask(sender):
            sender.tag().setContainer(self.pageContainer)
            self.pageContainer.updateAll()

        return VStack(items=[
            Button(ZStack(items=[
                Rect(color=Color.orange, cornerRadius=10),
                Label(tasks[i])
            ]), tag=pages[i], run=selectTask) for i in range(len(tasks))
        ], ratios=[1 / (len(tasks) * 2) for _ in range(len(tasks))])


class IntroDTPage(ZStack):
    def __init__(self):
        views = [
            Label("Decision Tree Description Here...")
        ]
        super().__init__(items=views)


class ExcerisePage(ZStack):
    def __init__(self):

        # Init Button Methods
        def updateTable(sender):
            self.setTable(Table(filePath="examples/" + sender.name, fontSize=20))
            self.updateContainers()

        def createTableCell(sender, index):
            if index != None:
                i, j = index // sender.cols, index % sender.cols
                column = sender.colNames[j]
                return ZStack(items=[
                    Rect(color=Color.steelBlue if index < sender.cols else Color.lightSteelBlue, border=3, cornerRadius=5, keywords="rect"),
                    Label(text=column if i == 0 else str(sender[column][sender.dataIndex[i - 1]]), fontSize=20, color=Color.white, keywords="label")
                ])

        # Declare Containers
        self.tableContainer = None
        self.treeContainer = None
        self.colSelContainer = None

        # Set Table
        self.setTable(Table(filePath="examples/zoo", createView=createTableCell, keywords="table"))
        files = hp.getFiles("examples", ext="csv")
        textboxManager = TextBoxManager(script=[
            ("Welcome to the Decision Tree Simulator", 0, 0),
            ("On the left hand side, we have the file explorer", -0.5, -0.5)
        ], stack=self)

        # Shift Buttons
        upShiftButton = self.createShiftButton(text="Up", tag=1)
        downShiftButton = self.createShiftButton(text="Down", tag=-1)

        # Build View
        views = [
            VStack(items=[

                # ===============================================================================================
                # TITLE
                # ===============================================================================================
                ZStack(items=[
                    Rect(color=Color.steelBlue),
                    Label("Decision Tree")
                ]),

                HStack(items=[
                    # ===============================================================================================
                    # FILE EXPLORER
                    # ===============================================================================================
                    # VStack(items=[
                    #     ZStack(items=[
                    #         Rect(color=Color.steelBlue, cornerRadius=10),
                    #         Button(view=Label("Files", fontSize=20), altView=Label("Clicked!!"), run=updateTable)
                    #     ])] + [
                    #     ZStack(items=[
                    #         Rect(color=Color.steelBlue, cornerRadius=10),
                    #         Button(view=Label(fileName.split(".")[0], fontSize=15), altView=Label("Clicked!!"), name=fileName, run=updateTable)
                    #     ]) for fileName in files
                    # ], ratios=[0.67 / (len(files) + 1)] * (len(files) + 1)),

                    VStack(items=[
                        # ===============================================================================================
                        # TABLE & MODEL
                        # ===============================================================================================
                        HStack(items=[
                            VStack(items=[
                                self.table,

                                # SHIFT BUTTONS
                                HStack(items=[
                                    upShiftButton,
                                    downShiftButton
                                ])
                            ], ratios=[0.9, 0.1]),
                            self.tree.getView()

                        ], ratios=[0.65, 0.35]),

                        # ===============================================================================================
                        # HYPERPARAMETERS
                        # ===============================================================================================
                        self.headerSelection

                    ], ratios=[0.9, 0.1])
                ])
            ], ratios=[0.08, 0.92]),

            # ===============================================================================================
            # TEXT BOX OVERLAY
            # ===============================================================================================
            textboxManager.createView()
        ]
        # print(view)
        super().__init__(items=views)

        # Define Containers
        self.tableContainer = self.table.container
        self.treeContainer = self.tree.getContainer()
        self.colSelContainer = self.headerSelection.container

    def setTable(self, table):
        self.table = table
        self.table.setContainer(self.tableContainer)

        def goForward(sender):
            # print("Forward:", sender, self.tree.hasChildren(), sender.tag)
            if self.tree.hasChildren():
                self.tree.go(index=sender.tag)
                self.selectedButton = None
                # self.treeContainer.updateAll()
                self.table = self.tree.getTable()
                self.table.setContainer(self.tableContainer)
                self.updateHeaderSelectionButtons()
                self.updateContainers()

        def goBack(sender):
            # print("Back:", sender, not self.tree.isRoot())
            if not self.tree.isRoot():
                self.tree.goBack()
                self.table = self.tree.getTable()
                self.table.setContainer(self.tableContainer)
                self.updateHeaderSelectionButtons()
                self.updateContainers()
                # print("Final Container:")
                # print(self.treeContainer)
            # self.tree.move(view)
            # self.treeContainer.updateAll()

        self.tree = DecisionTree(table, backMethod=goBack, goMethod=goForward)
        self.tree.getView().setContainer(self.treeContainer)
        self.selectedButton = None

        def selectColumn(sender):
            self.selectedButton = sender if sender.isOn else None
            self.table.selectColumn(value=sender.isOn, index=sender.tag)
            rect = sender.getCousin(0)
            if sender.isOn:
                self.tree.add(column=self.tree.getColName(sender.tag), backMethod=goBack, goMethod=goForward)
                rect.strokeColor = Color.white
            else:
                self.tree.remove()
                rect.strokeColor = Color.gray
            self.treeContainer.updateAll()
            self.updateHeaderSelectionButtons()

        columns = self.table.data.columns
        self.headerSelection = HStack(items=[
            ZStack(items=[
                Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
                Button(view=Label(columns[i], fontSize=25, color=Color.white),
                       toggleView=Label(columns[i], fontSize=18, color=Color.black),
                       #    altView=Label("{} CLK".format(columns[i]), fontSize=15),
                       isOn=False, tag=i, run=selectColumn)
            ]) for i in range(1, len(columns))

        ])
        self.headerSelection.setContainer(self.colSelContainer)
        self.updateHeaderSelectionButtons()

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

    def createShiftButton(self, text, tag):
        def clickShift(sender):
            self.table.shift(dy=sender.tag)
            self.tree.updateTableColor()
            if self.selectedButton:
                self.table.selectColumn(value=True, index=self.selectedButton.tag)
            self.table.updateAll()
        return Button(view=ZStack(items=[
            Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3),
            Label(text=text, fontSize=20, color=Color.black)
        ]), lockedWidth=80, lockedHeight=45, run=clickShift, tag=tag)


class TextBoxManager:
    def __init__(self, script, stack):
        self.script = script
        self.index = 0
        self.stack = stack

    def createView(self):
        text, dx, dy = self.script[self.index]
        return Button(view=ZStack(items=[
            Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
            Label(text=text, color=Color.black, fontSize=15)
        ]), run=self.pressTextBox, dx=dx, dy=dy, lockedWidth=250, lockedHeight=100)

    def pressTextBox(self, sender):
        self.stack.popView()
        self.index += 1
        if self.index < len(self.script):
            self.stack.addView(self.createView())
        self.stack.updateAll()
