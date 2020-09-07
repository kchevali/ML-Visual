from graphics import *
from decisionTree import DecisionTree
from table import Table
from random import shuffle
import helper as hp
from math import sin, cos, pi
import subprocess


class Controller:
    def __init__(self, filePath=None, partition=True, textboxScript=None, textboxStack=None):
        self.filePath = filePath
        self.fileName = None
        if self.filePath:
            self.fileName = self.filePath.split("/")[-1].split(".")[0]
            # TABLES
            if partition:
                self.mainTable = Table(filePath=self.filePath)
                self.trainTable, self.finalTestTable = self.mainTable.partition()
                self.table, self.localTestTable = self.trainTable.partition()
            else:
                self.table = Table(filePath=filePath)
            self.tableView = None

            # TREE
            self.tree = DecisionTree(self.table)
            self.classColors = {}
            self.treeView = None
            self.bagIndex = 0
            self.treeList = []

        # Codes
        self.codes = [
            Code("model = DecisionTreeClassifier()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]

        # Bagging
        self.textboxScript = textboxScript
        self.textboxStack = textboxStack
        self.textboxIndex = 0

    # ============================================================================
    # CREATE VIEWS
    # ============================================================================

    def createFileNameView(self):
        return Label("File: " + self.fileName, fontSize=18, dx=-1)

    def createTableView(self, **args):
        tableViewContainer = self.tableView.container if self.tableView != None else None
        self.tableView = self.table.createView(createCell=self.createTableCell, **args)
        if tableViewContainer != None:
            self.tableView.setContainer(tableViewContainer)
        self.selectedColumns = [False for _ in range(self.tableView.cols)]
        if self.classColors:
            self.colorTableTargets()
        return self.tableView

    def createFileExplorerView(self):
        files = hp.getFiles("examples", ".csv")
        self.fileExplorer = ZStack([
            Rect(Color.backgroundColor, border=0),
            VStack([
                ZStack([
                    Rect(color=Color.steelBlue, cornerRadius=10),
                    Label("Files", fontSize=20)
                ])] + [
                Button([
                    Rect(color=Color.steelBlue, cornerRadius=10),
                    Label(fileName.split(".")[0], fontSize=15)
                ], name=fileName, lockedWidth=150) for fileName in files
            ], ratios=[0.7 / (len(files) + 1)] * (len(files) + 1))

        ], lockedWidth=350, lockedHeight=600)
        return self.fileExplorer

    def createHeaderButtons(self):
        columns = self.table.columns
        self.headerSelection = HStack([
            ZStack([
                Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
                Button(Label(columns[i], fontSize=25, color=Color.white),
                       isOn=False, tag=i, run=self.selectColumn)
            ]) for i in range(1, len(columns))

        ])
        return self.headerSelection

    def createTreeRoomView(self):
        self.classColors = {}
        for i in range(self.table.classCount):
            self.classColors[self.table.classes[i]] = Color.calmColor(i / self.table.classCount)
        self.colorTableTargets()
        treeViewContainer = self.treeView.container if self.treeView != None else None
        self.treeView = self.createDotViews(self.tree.current)
        if treeViewContainer != None:
            self.treeView.setContainer(treeViewContainer)
        self.treeBranch = Branch(view=self.treeView, disjoint=Container())
        return self.treeView

    def createCodeTitle(self):
        self.treeCountLabel = Label("Add Trees")
        self.codeAccuracyLabel = Label("Accuracy: --")
        self.runButtonRect = Rect(color=Color.gray, cornerRadius=10)
        self.runButton = Button([
            self.runButtonRect,
            Label("Run")
        ], hideAllContainers=True, lockedWidth=240, run=self.runTrees, isDisabled=True)

        self.codingTitle = HStack([
            self.runButton,
            self.codeAccuracyLabel,
            Button([
                Rect(color=Color.steelBlue, cornerRadius=10),
                self.treeCountLabel
            ], run=self.incTrees)
        ], ratios=[0.5, 0.25, 0.25])
        return self.codingTitle

    def createCodeView(self):
        codeViews = [
            ZStack([
                Rect(color=Color.steelBlue, keywords="rect", cornerRadius=10),
                Label(code.label, keywords="label")
            ], isDraggable=True, tag=code, keywords="codeStack", lockedWidth=200, lockedHeight=60, hideAllContainers=True)for code in self.codes
        ]
        shuffle(codeViews)

        self.codeLabelStack = VStack(codeViews, keywords="question")
        return self.codeLabelStack

    def createOpenSpreadsheetView(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Excel")
        ], run=self.openFile, tag=self.filePath + ".csv", lockedWidth=200)

    def createCodeFileView(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Code")
        ], run=self.openFile, tag="assets/treeExample.py", lockedWidth=200)

    def createTreeListView(self):
        self.totalScoreLabel = Label("Total: [0%]", fontSize=20)
        self.totalScore = ZStack([
            Rect(color=Color.steelBlue),
            self.totalScoreLabel
        ], lockedHeight=50, dy=1)
        self.bag = VStack([None] * 9 + [
            self.totalScore,
            Button([
                Rect(color=Color.steelBlue, cornerRadius=10),
                Label("Save Tree")
            ], run=self.saveTree, lockedHeight=50, dy=1)
        ])
        return self.bag

    def createInfoDTViews(self):
        buttons = []
        for label, path in [
            ("Generatation", "assets/GenerateDecisionTree.pdf"),
            ("Improvement", "assets/ImproveDecisionTree.pdf")
        ]:
            def setView(sender):
                labelView = sender.keyDown("label")
                if sender.isAlt():
                    labelView.setFont("Opening File...")
                else:
                    labelView.setFont("Click here to learn more about: " + label)

            buttons.append(Button(
                Label("", dx=-1, offsetX=10, keywords="label"),
                tag=path,
                run=self.openFile,
                setViewMethod=setView
            ))

        return VStack(
            [Label("More Information Below:", fontSize=48, dx=-1)] + buttons + [None] * 8, hideAllContainers=False
        )

    # ============================================================================
    # BUTTON METHODS
    # ============================================================================

    def selectColumn(self, sender):
        self.selectTableViewColumn(value=sender.isOn, column=sender.tag)
        rect = sender.getCousin(0)
        if sender.isOn:
            self.splitTree(column=self.tree.getColName(sender.tag))
            rect.strokeColor = Color.white
        else:
            self.tree.remove()
            rect.strokeColor = Color.gray
        self.treeBranch.getContainer().updateAll()
        self.updateHeaderSelectionButtons()
        # self.tableView.container.updateAll()

    def goForward(self, sender):
        if self.tree.hasChildren():
            self.goToIndexTree(index=sender.tag)
            self.table = self.tree.getTable()
            self.createTableView()
            self.updateHeaderSelectionButtons()
            self.treeBranch.getContainer().updateAll()
            self.tableView.container.updateAll()
            # self.updateContainers()

    def goBack(self, sender):
        if not self.tree.isRoot():
            self.goBackTree()
            self.table = self.tree.getTable()
            self.createTableView()
            self.updateHeaderSelectionButtons()
            self.treeBranch.getContainer().updateAll()
            self.tableView.container.updateAll()

    def saveTree(self, sender):
        self.treeList.append(self.tree)
        accuracy = round(self.tree.test(self.localTestTable) * 100)

        self.tree.modelTest(self.finalTestTable)

        bagItem = ZStack([
            Rect(color=Color.steelBlue),
            Label(text="T:{} [{}%]".format(len(self.treeList), accuracy), fontSize=20)
        ], dy=1)
        bagItem.setContainer(self.bag[self.bagIndex])
        self.bagIndex += 1

        self.totalScoreLabel.setFont(text="Total: [{}%]".format(round(100 * self.getTotalScore(finalTestData=self.finalTestTable.data, predictMethod=self.tree.predict))))
        self.table, self.localTestTable = self.trainTable.partition()

        self.tree = DecisionTree(self.table)
        self.createTreeRoomView()
        self.createTableView()

        self.bag.updateAll()
        self.treeView.container.updateAll()
        self.tableView.container.updateAll()
        self.updateHeaderSelectionButtons()

    def incTrees(self, sender):
        if len(self.treeList) >= 100:
            self.treeList = []
            self.treeCountLabel.setFont("Add Trees")

        else:
            for _ in range(1):
                train, test = self.trainTable.partition()
                self.treeList.append(DecisionTree(train))
            self.treeCountLabel.setFont("Trees: {}".format(len(self.treeList)))

    def runTrees(self, sender):
        self.codeAccuracyLabel.setFont(text="Accuracy: {}%".format(round(100 * self.getTotalScore(finalTestData=self.finalTestTable.encodedData, predictMethod=self.tree.modelPredict))))
        self.codeAccuracyLabel.container.updateAll()

    def pressTextBox(self, sender):
        self.textboxStack.popView()
        self.textboxIndex += 1
        if self.textboxIndex < len(self.textboxScript):
            self.textboxStack.addView(self.createNextTextbox())
        self.textboxStack.updateAll()

    def openFile(self, sender):
        subprocess.run(['open', sender.tag], check=True)

    # ============================================================================
    # CHANGE VIEWS
    # ============================================================================

    def selectTableViewColumn(self, value, column):
        self.selectedColumns[column] = value
        for i in range(1, self.tableView.rows):
            view = self.tableView.getView(i * self.tableView.cols + column)
            if view != None:
                view.keyDown("rect").color = Color.steelBlue if value else Color.lightSteelBlue

    def colorTableTargets(self):
        index = self.tableView.cols
        startRow = self.tableView.ci
        for item in self.table.targetCol:
            if startRow > 0:
                startRow -= 1
                continue
            if index >= self.tableView.length:
                break
            rect = self.tableView.getView(index).keyDown("rect")
            rect.color = self.classColors[item]
            rect.isHidden = False
            index += self.tableView.cols

    def createNextTextbox(self):
        text, dx, dy = self.textboxScript[self.textboxIndex]
        return Button([
            Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
            Label(text=text, color=Color.black, fontSize=25)
        ], run=self.pressTextBox, dx=dx, dy=dy, lockedWidth=450, lockedHeight=200, hideAllContainers=True)

    def shiftTable(self, dy):
        self.tableView.shift(dy=dy)
        self.colorTableTargets()
        self.updateHeaderSelectionButtons()
        self.tableView.updateAll()

    def updateHeaderSelectionButtons(self):
        columns = self.table.columns
        for c in self.headerSelection.containers:
            rect = c.view.getView(0)
            button = c.view.getView(1)
            column = columns[button.tag]

            if self.tree.isParentColumn(column):
                rect.strokeColor = Color.gray
                button.isDisabled = True
                button.setOn(isOn=False)
            else:
                button.isDisabled = False
                if self.tree.isCurrentColumn(column):
                    rect.strokeColor = Color.yellow
                    button.setOn(isOn=True)
                else:
                    rect.strokeColor = Color.darkGray
                    button.setOn(isOn=False)
                    self.selectTableViewColumn(value=False, column=button.tag)

    def createTableCell(self, tableView, index):
        if index != None:
            i, j = index // tableView.cols, index % tableView.cols
            if self.tableView != None and self.selectedColumns[j]:
                self.selectTableViewColumn(True, j)

            column = self.table.colNames[j]
            return ZStack([
                Rect(color=Color.steelBlue if index < tableView.cols else Color.lightSteelBlue, border=3, cornerRadius=5, keywords="rect"),
                Label(text=column if i == 0 else str(self.table[column][self.table.dataIndex[i - 1]]), fontSize=20, color=Color.white, keywords="label")
            ])

    def createDotViews(self, treeNode):
        if treeNode.table.dataRows > 0:
            trig = 2.0 * pi / treeNode.table.dataRows

        items = []
        i = 0
        for index, row in treeNode.table.iterrows():
            items.append(Ellipse(color=self.classColors[row[treeNode.table.targetName]],
                                 strokeColor=Color.red, strokeWidth=2,
                                 dx=0.5 * cos(trig * i) if treeNode.table.dataRows > 1 else 0.0,
                                 dy=0.5 * sin(trig * i) if treeNode.table.dataRows > 1 else 0.0,
                                 border=0,
                                 lockedWidth=20, lockedHeight=20
                                 ))
            i += 1

        if treeNode.parent != None:
            items.append(Label(text="{}:{}".format(treeNode.parent.column, treeNode.value), fontSize=20, color=Color.white, dx=-0.95, dy=-1))
        return ZStack(items=items, keywords="dotStack", limit=150)

    def splitTree(self, column):
        self.tree.add(column=column)
        container = self.treeBranch.getContainer()
        prevStack = type(self.treeBranch.disjoint.keyUp("div"))
        stack = VStack if prevStack != VStack else HStack
        label = [Button(Label(text="{}:{}".format(self.tree.getParentColumn(), self.tree.getValue()), fontSize=20,
                                   color=Color.white, dx=-0.95, dy=-1), run=self.goBack)] if self.tree.getParent() != None else []
        treeChildren = self.tree.getChildren()
        totalClassCount = len(treeChildren)
        for child in treeChildren:
            totalClassCount += child.table.classCount

        self.treeView = ZStack(items=label + [
            stack(items=[
                Button(self.createDotViews(self.tree.getChild(i)), run=self.goForward, tag=i) for i in range(len(treeChildren))
            ], ratios=[
                (child.table.classCount + 1) / totalClassCount for child in treeChildren
            ], border=20 if self.tree.getParent() != None else 0, keywords="div")
        ], keywords=["z"])
        self.treeBranch.setView(view=self.treeView)

    def removeNodeTree(self):
        self.tree.remove()
        self.treeView = self.createDotViews(self.tree.current)
        self.treeBranch.setView(view=self.treeView)

    def goBackTree(self):
        self.tree.goBack()
        self.treeView = self.treeBranch.disjoint.keyUp("z")
        self.treeBranch.move(self.treeView)

    def goToIndexTree(self, index):
        self.tree.go(index=index)
        container = self.treeBranch.view.keyDown("div")[index]
        stack = container.keyDown("z")
        self.treeView = stack if stack != None else container.keyDown("dotStack")
        self.treeBranch.move(self.treeView)

    # ============================================================================
    # OTHER
    # ============================================================================

    def getTotalScore(self, finalTestData, predictMethod):
        if not self.treeList:
            return 0.0
        correct = 0
        for index, row in finalTestData.iterrows():
            answers = {"": -1}
            bestAnswer = ""
            for tree in self.treeList:
                answer = predictMethod(row)
                if answer not in answers:
                    answers[answer] = 1
                else:
                    answers[answer] += 1
                if answers[answer] > answers[bestAnswer]:
                    bestAnswer = answer
            if bestAnswer == row[self.finalTestTable.targetName]:
                correct += 1
        return correct / self.finalTestTable.dataRows


class Code:

    def __init__(self, line, label, order):
        self.line = line
        self.label = label
        self.order = order


class Branch:
    def __init__(self, view, disjoint):
        self.disjoint = disjoint  # Container
        self.view = view  # Noncontainer

    def getContainer(self):
        return self.view.container

    def setView(self, view):
        view.setContainer(container=self.view.container)
        self.view = view

    def move(self, nextView):
        if nextView != None:
            nextDisjoint = nextView.container
            nextView.setContainer(container=self.view.container)
            self.view.setContainer(container=self.disjoint)
            self.disjoint = nextDisjoint
            self.view = nextView
