from graphics import *
import helper as hp
import viewHelper as vp
from table import Table
from models import *
from random import shuffle
from math import sin, cos, pi
from elements import *


modelTitle = ""


def createMouseDebug():
    return ZStack([
        Rect(color=Color.white, keywords="mRect", border=0),
        Label(text="", fontSize=15, color=Color.black, keywords="text")
    ], lockedWidth=80, lockedHeight=20, dx=-1, dy=1)


# =====================================================================
# Start Up Classes
# =====================================================================

class DefaultPage(ZStack):
    def __init__(self):
        items = [
            Color(Color.backgroundColor)
        ]
        super().__init__(items)


class SimplePage(ZStack):
    def __init__(self):
        def createView(sender, index):
            return Label(str(index))

        items = [
            Table(filePath="examples/decisionTree/shape", createView=createView)
        ]
        super().__init__(items)


class MainPage(ZStack):
    def __init__(self):
        items = [
            MenuPage()
        ]
        super().__init__(items)

    def canDragView(self, view, container):
        canDrag = False
        for c in self.containers:
            canDrag = canDrag or c.view.canDragView(view=view, container=container)
        return canDrag

    def draggedView(self, view):
        for c in self.containers:
            c.view.draggedView(view=view)

    def hoverMouse(self, x, y):
        for c in self.containers:
            c.view.hoverMouse(x, y)

    def update(self):
        for c in self.containers:
            c.view.update()


class ModelPage(ZStack):
    def __init__(self, content, title, pages=[], includeTaskList=True):

        self.content = content
        self.title = title
        self.pages = pages
        self.includeTaskList = includeTaskList

        self.content.modelPage = self

        items = [
            VStack([
                createLabel(self.title, views=[Button(Label("<"), run=self.replaceSelf, tag=MenuPage, lockedWidth=40, lockedHeight=40, dx=-1, offsetX=20) if type(self) != MenuPage else None]),
                HStack([
                    self.createTaskList(),
                    self.content
                ], ratios=[0.15, 0.85]) if self.includeTaskList else self.content
            ], ratios=[0.08, 0.92])

        ]
        super().__init__(items)

    def createTaskList(self):
        return VStack([
            createButton(text=task, color=Color.orange, tag=page, run=self.replaceContent) for task, page in self.pages
        ], ratios=[1 / (len(self.pages) * 2) for _ in range(len(self.pages))])

    def canDragView(self, view, container):
        return self.content.canDragView(view=view, container=container)

    def draggedView(self, view):
        return self.content.draggedView(view=view)

    def scrollUp(self):
        self.content.scrollUp()

    def scrollDown(self):
        self.content.scrollDown()

    def replaceContent(self, sender):
        self.content = sender.tag().replaceView(self.content)
        self.content.container.updateAll()

    def replaceSelf(self, sender):
        global modelTitle
        if sender.tag != None:
            self.content = sender.tag().replaceView(self)
            modelTitle = self.content.title
            self.content.container.updateAll()

    def hoverMouse(self, x, y):
        self.content.hoverMouse(x, y)

    def update(self):
        self.content.update()


class MenuPage(ModelPage):
    def __init__(self):
        content = HStack([
            None,
            VStack([
                createLabel(text="Classical Models", color=Color.green),
                self.createMenuButton(text="KNN", color=Color.red, tag=self.createKNN),
                self.createMenuButton(text="Linear Regression", color=Color.red, tag=self.createLinear),
                self.createMenuButton(text="Logistic Regression", color=Color.red, tag=self.createLogistic),
                None
            ], hideAllContainers=True),
            VStack([
                createLabel("Modern Models", color=Color.green),
                self.createMenuButton(text="Decision Tree", color=Color.blue, tag=self.createDecisionTree),
                self.createMenuButton(text="SVM", color=Color.gray),
                self.createMenuButton(text="Neural Networks", color=Color.gray),
                None
            ], hideAllContainers=True),
            None
        ], ratios=[0.15, 0.35, 0.35, 0.15])
        super().__init__(content=content, title="Select Machine Learning Model", includeTaskList=False)

    def createMenuButton(self, text, color, tag=None):
        return createButton(text=text, color=color, tag=tag, run=self.replaceSelf, hideAllContainers=True)

    def createDecisionTree(self):
        return ModelPage(content=IntroDTPage(), title="Decision Tree",
                         pages=[
            ("Intro", IntroDTPage),
            ("Example", ExampleDTPage),
            ("Improve", ExceriseDTPage),
            ("Coding", CodingDTPage),
            ("More Info", InfoDTPage)
        ])

    def createKNN(self):
        return ModelPage(content=ExampleKNNPage(), title="KNN",
                         pages=[
            ("Intro", IntroKNNPage),
            ("Example", ExampleKNNPage),
            ("Coding", CodingKNNPage),
            ("More Info", InfoKNNPage)
        ])

    def createLinear(self):
        return ModelPage(content=ExampleLinearPage(), title="Linear Regression",
                         pages=[
            ("Intro", IntroLinearPage),
            ("Example", ExampleLinearPage),
            ("Quadratic", QuadLinearPage),
            ("Coding", CodingLinearPage),
            ("More Info", InfoLinearPage)
        ])

    def createLogistic(self):
        return ModelPage(content=ExampleLogisticPage(), title="Logistic Regression",
                         pages=[
            ("Intro", IntroLogisticPage),
            ("Example", ExampleLogisticPage)
            # ("Coding", CodingLogisticPage),
            # ("More Info", InfoLogisticPage)
        ])

# =====================================================================
# BASE PAGE CLASSES
# =====================================================================


class TextBoxPage(ZStack):

    def __init__(self, textboxScript=None, textboxAudioPath=None, **kwargs):
        super().__init__(**kwargs)
        self.textboxScript = textboxScript
        self.textboxIndex = 0
        self.textboxAudioPath = textboxAudioPath

    def pressTextBox(self, sender):
        self.popView()
        self.textboxIndex += 1
        if self.textboxIndex < len(self.textboxScript):
            self.addView(self.createNextTextbox())
        self.updateAll()

    def createNextTextbox(self):
        text, dx, dy = self.textboxScript[self.textboxIndex]
        soundName = self.textboxAudioPath + str(self.textboxIndex + 1) if self.textboxAudioPath != None and self.textboxIndex != None else None
        return ZStack([
            Button([
                Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
                Label(text=text, color=Color.black, fontSize=25)
            ], run=self.pressTextBox),
            Button(Image(imageName="audio.png", lockedWidth=50, lockedHeight=50), soundName=soundName, dx=1, dy=1, offsetX=-20, offsetY=-20, lockedWidth=60, lockedHeight=60)
        ], dx=dx, dy=dy, lockedWidth=450, lockedHeight=200, hideAllContainers=True)


class IntroPage(ZStack):

    def __init__(self, description, **kwargs):
        views = [
            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),
            Label(description)
        ]
        super().__init__(views, **kwargs)


class TablePage(TextBoxPage):
    def __init__(self, filePath, partition=True, **kwargs):
        super().__init__(**kwargs)
        self.filePath = filePath
        self.fileName = self.filePath.split("/")[-1].split(".")[0] if self.filePath else None
        self.partition = partition
        if self.partition:
            self.mainTable = Table(filePath=self.filePath)
            self.trainTable, self.finalTestTable = self.mainTable.partition()
            self.table, self.localTestTable = self.trainTable.partition()
        else:
            self.table = Table(filePath=self.filePath)
        self.tableView = None
        self.classColors = {}
        for i in range(self.table.classCount):
            self.classColors[self.table.classes[i]] = Color.calmColor(i / self.table.classCount)

    def createTableView(self, **kwargs):
        prevContainer = self.tableView.container if self.tableView != None else None
        self.tableView = self.table.createView(createCell=self.createTableCell, **kwargs)
        if prevContainer != None:
            self.tableView.setContainer(prevContainer)
        self.selectedColumns = [False for _ in range(self.tableView.cols)]
        if self.classColors:
            self.colorTableTargets()
        return self.tableView

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

    def selectTableViewColumn(self, value, column):
        self.selectedColumns[column] = value
        for i in range(1, self.tableView.rows):
            view = self.tableView.getView(i * self.tableView.cols + column)
            if view != None:
                view.keyDown("rect").color = Color.steelBlue if value else Color.lightSteelBlue

    def shiftTable(self, dy):
        self.tableView.shift(dy=dy)
        self.colorTableTargets()
        self.updateHeaderSelectionButtons()
        self.tableView.updateAll()

    def scrollUp(self):
        self.shiftTable(dy=1)

    def scrollDown(self):
        self.shiftTable(dy=-1)


class MLPage(TablePage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = []
        self.mainModel = None
        self.modelView = None
        if self.filePath:
            self.modelMousePoints = []

    def createLegendView(self, separateByTarget=True):
        if separateByTarget:
            legendItems = [Label("Legend", fontSize=30, color=Color.white)] + [
                Label(self.table.classes[i], fontSize=30, color=self.classColors[self.table.classes[i]]) for i in range(self.table.classCount)
            ]

            return ZStack([
                Rect(color=Color.backgroundColor, strokeWidth=3, strokeColor=Color.steelBlue, cornerRadius=10),
                VStack(legendItems, ratios=[0.8 / len(legendItems) for item in legendItems], offsetY=10, hideAllContainers=True)
            ], lockedWidth=150, lockedHeight=180, dx=0.8, dy=-0.8, hideAllContainers=True)
        return None

    def createDots(self, separateByTarget=True):
        items = []
        for index, row in self.table.normalized.iterrows():
            items.append(Ellipse(color=self.classColors[row[self.table.targetName]] if separateByTarget else Color.steelBlue,
                                 strokeColor=Color.white, strokeWidth=3, dx=row[self.table.first()], dy=row[self.table.second()], lockedWidth=15, lockedHeight=15))
        return items

    def createGraphView(self, separateByTarget=True, clickPoint=None):

        self.legend = self.createLegendView(separateByTarget)

        self.modelView = ZStack([
            Button(self.createDots(separateByTarget), limit=1000, run=clickPoint),
            self.legend,
            self.createIncButton(dx=-1, dy=1)
        ], limit=100)

        axises = [ZStack([
            Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
            Label(self.table.columns[i], fontSize=25, color=Color.white, angle=90 * (2 - i))
        ], tag=i)for i in range(1, len(self.table.columns))]

        self.modelGraph = HStack([
            VStack([
                axises[0],
                None,
            ], ratios=[0.9, 0.1]),
            VStack([
                self.modelView,
                axises[1]
            ], ratios=[0.9, 0.1])
        ], ratios=[0.08, 0.92])
        return self.modelGraph

    def getTotalScore(self):
        if not self.models:
            return 0.0
        correct = 0
        for index, row in self.finalTestTable.iterrows():
            answers = {"": -1}
            bestAnswer = ""
            for model in self.models:
                answer = model.predict(row)
                if answer not in answers:
                    answers[answer] = 1
                else:
                    answers[answer] += 1
                if answers[answer] > answers[bestAnswer]:
                    bestAnswer = answer
            if bestAnswer == row[self.finalTestTable.targetName]:
                correct += 1
        return correct / self.finalTestTable.dataRows

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

    def selectColumn(self, sender):
        self.selectTableViewColumn(value=sender.isOn, column=sender.tag)
        rect = sender.getCousin(0)
        if sender.isOn:
            self.splitTree(column=self.mainModel.getColName(sender.tag))
            rect.strokeColor = Color.white
        else:
            self.mainModel.remove()
            rect.strokeColor = Color.gray
        self.modelBranch.getContainer().updateAll()
        self.updateHeaderSelectionButtons()
        # self.tableView.container.updateAll()

    def updateHeaderSelectionButtons(self):
        columns = self.table.columns
        for c in self.headerSelection.containers:
            rect = c.view.getView(0)
            button = c.view.getView(1)
            column = columns[button.tag]

            if self.mainModel.isParentColumn(column):
                rect.strokeColor = Color.gray
                button.isDisabled = True
                button.setOn(isOn=False)
            else:
                button.isDisabled = False
                if self.mainModel.isCurrentColumn(column):
                    rect.strokeColor = Color.yellow
                    button.setOn(isOn=True)
                else:
                    rect.strokeColor = Color.darkGray
                    button.setOn(isOn=False)
                    self.selectTableViewColumn(value=False, column=button.tag)

    def createIncButton(self, **kwargs):
        pass

    def incMethod(self, sender):
        pass

    def createLines(self, color, errorOffset, eqOffset, **kwargs):
        lines = Lines(color=color)
        errorLabel = Label("Error: --", color=color, offsetY=errorOffset, **kwargs)
        eqLabel = Label("Y=--", color=color, offsetY=eqOffset, **kwargs)
        self.modelView.addAllViews(lines, errorLabel, eqLabel)
        return lines, errorLabel, eqLabel

    def clickSlope(self, sender):
        self.mainModel.setSize(self.modelView)
        points = self.mainModel.addPoint((sender.lastClickX, sender.lastClickY))

        if points != None:
            self.modelLine.points = points
            self.modelError.setFont(text="Error: {}".format(round(self.mainModel.getError(), 4)))
            # self.modelEq.setFont(text=self.mainModel.getEqString())
        else:
            self.modelLine.points = []

    def createAddCompButton(self):
        self.addCompButton = Button([
            Rect(color=Color.backgroundColor, cornerRadius=10, strokeColor=Color.steelBlue, strokeWidth=3),
            Label("ML Results")
        ], lockedWidth=150, lockedHeight=80, dx=1, dy=-1, offsetX=-10, offsetY=10, run=self.startComp)
        return self.addCompButton

    def startComp(self, sender):
        self.compTrainComplete = False
        self.compLine = Lines(color=Color.blue)
        self.modelView.addView(self.compLine)

    def hoverMouse(self, x, y):
        if self.hoverEnabled:
            self.mainModel.setSize(self.modelView)
            points = self.mainModel.addPoint((x, y), storePoint=False)
            if points != None:
                self.modelLine.points = points
                self.modelError.setFont(text="Error: {}".format(round(self.mainModel.getError(), 4)))
                # self.modelEq.setFont(text=self.mainModel.getEqString())


class DTPage(MLPage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mainModel = DecisionTree(table=self.table)
        self.models.append(self.mainModel)
        if self.filePath:
            self.bagIndex = 0

    def createTreeRoomView(self):
        treeViewContainer = self.modelView.container if self.modelView != None else None
        self.modelView = self.createDotViews(self.mainModel.current)
        if treeViewContainer != None:
            self.modelView.setContainer(treeViewContainer)
        self.modelBranch = Branch(view=self.modelView, disjoint=Container())
        return self.modelView

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

    def goForward(self, sender):
        if self.mainModel.hasChildren():
            self.goToIndexTree(index=sender.tag)
            self.table = self.mainModel.getTable()
            self.createTableView()
            self.updateHeaderSelectionButtons()
            self.modelBranch.getContainer().updateAll()
            self.tableView.container.updateAll()
            # self.updateContainers()

    def goBack(self, sender):
        if not self.mainModel.isRoot():
            self.goBackTree()
            self.table = self.mainModel.getTable()
            self.createTableView()
            self.updateHeaderSelectionButtons()
            self.modelBranch.getContainer().updateAll()
            self.tableView.container.updateAll()

    def saveTree(self, sender):
        accuracy = round(self.mainModel.test(self.localTestTable) * 100)
        self.mainModel.modelTest(self.finalTestTable)

        bagItem = ZStack([
            Rect(color=Color.steelBlue),
            Label(text="T:{} [{}%]".format(len(self.models), accuracy), fontSize=20)
        ], dy=1)
        bagItem.setContainer(self.bag[self.bagIndex])
        self.bagIndex += 1

        self.totalScoreLabel.setFont(text="Total: [{}%]".format(round(100 * self.getTotalScore())))
        self.table, self.localTestTable = self.trainTable.partition()

        self.mainModel = DecisionTree(table=self.table)
        self.models.append(self.mainModel)
        self.createTreeRoomView()
        self.createTableView()

        self.bag.updateAll()
        self.modelView.container.updateAll()
        self.tableView.container.updateAll()
        self.updateHeaderSelectionButtons()

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
        self.mainModel.add(column=column)
        container = self.modelBranch.getContainer()
        prevStack = type(self.modelBranch.disjoint.keyUp("div"))
        stack = VStack if prevStack != VStack else HStack
        label = [Button(Label(text="{}:{}".format(self.mainModel.getParentColumn(), self.mainModel.getValue()), fontSize=20,
                              color=Color.white, dx=-0.95, dy=-1), run=self.goBack)] if self.mainModel.getParent() != None else []
        treeChildren = self.mainModel.getChildren()
        totalClassCount = len(treeChildren)
        for child in treeChildren:
            totalClassCount += child.table.classCount

        self.modelView = ZStack(items=label + [
            stack(items=[
                Button(self.createDotViews(self.mainModel.getChild(i)), run=self.goForward, tag=i) for i in range(len(treeChildren))
            ], ratios=[
                (child.table.classCount + 1) / totalClassCount for child in treeChildren
            ], border=20 if self.mainModel.getParent() != None else 0, keywords="div")
        ], keywords=["z"])
        self.modelBranch.setView(view=self.modelView)

    def removeNodeTree(self):
        self.mainModel.remove()
        self.modelView = self.createDotViews(self.mainModel.current)
        self.modelBranch.setView(view=self.modelView)

    def goBackTree(self):
        self.mainModel.goBack()
        self.modelView = self.modelBranch.disjoint.keyUp("z")
        self.modelBranch.move(self.modelView)

    def goToIndexTree(self, index):
        self.mainModel.go(index=index)
        container = self.modelBranch.view.keyDown("div")[index]
        stack = container.keyDown("z")
        self.modelView = stack if stack != None else container.keyDown("dotStack")
        self.modelBranch.move(self.modelView)


class KNNPage(MLPage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mainModel = KNN(table=self.table)
        self.models.append(self.mainModel)

    def incMethod(self, sender):
        self.mainModel.k += 2
        if self.mainModel.k > 10:
            self.mainModel.k = 1
        self.codingAddLabel.setFont("K: {}".format(self.mainModel.k))
        for rect in self.modelMousePoints:
            rect.color = self.classColors[self.mainModel.predictPoint(*rect.tag)]

    def createIncButton(self, **kwargs):
        self.codingAddLabel = Label("K: {}".format(self.mainModel.k))
        return Button([
            Rect(color=Color.backgroundColor, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
            self.codingAddLabel
        ], run=self.incMethod, lockedWidth=130, lockedHeight=80, **kwargs)


class LinearPage(MLPage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mainModel = Linear(table=self.table)
        self.models.append(self.mainModel)

    def createIncButton(self, **kwargs):
        pass
        # self.codingAddLabel = Label("Power: {}".format(self.mainModel.n))
        # return Button([
        #     Rect(color=Color.backgroundColor, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
        #     self.codingAddLabel
        # ], run=self.incMethod, lockedWidth=130, lockedHeight=80, **kwargs)

    def incMethod(self, sender):
        pass
        # self.mainModel.n = (self.mainModel.n & 1) + 1
        # self.mainModel.reset()
        # self.codingAddLabel.setFont("Power: {}".format(self.mainModel.n))
        # self.modelLine.points = []


class LogisticPage(MLPage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mainModel = Logistic(table=self.table)
        self.models.append(self.mainModel)


class CodingPage(MLPage):
    def __init__(self, codes, codingAddString, codingFilePath, codingIncMethod, examplePath, **kwargs):
        self.codes = codes
        self.codingAddString = codingAddString
        self.codingFilePath = codingFilePath
        self.codingIncMethod = codingIncMethod
        self.codingExamplePath = examplePath

        super().__init__(textboxScript=[
            ("Welcome to the Coding Tutorial!", 0, 0),
            (["On this page, we will show the basics on how to",
                "run a " + modelTitle + " on Python"], 0, 0),
            (["Lets begin by dragging the code labels on the",
                "left column to the correct spots on right"], 0.5, 0),
            ("Can you figure out the correct order?", 0, 0),
            (["Once you successfully set the code blocks,",
                "add some trees and run your code!"], 0, 0)
        ], **kwargs)

        items = [
            VStack([
                self.createCodingHeader(),
                self.createCodingTable(),
                self.createCodingOptions()
            ], ratios=[0.1, 0.8, 0.1]),
            self.createNextTextbox()

        ]
        ZStack.__init__(self, items=items, hoverEnabled=False)

    def createCodingHeader(self):
        self.codingRunRect = Rect(color=Color.gray, cornerRadius=10)
        self.codingRunButton = Button([
            self.codingRunRect,
            Label("Run")
        ], hideAllContainers=True, lockedWidth=240, run=self.runCodingTest, isDisabled=True)

        self.codingAccLabel = Label("Accuracy: --")
        self.codingAddLabel = Label(self.codingAddString)
        self.codingHeader = HStack([
            self.codingRunButton,
            self.codingAccLabel,
            Button([
                Rect(color=Color.steelBlue, cornerRadius=10),
                self.codingAddLabel
            ], run=self.codingIncMethod)
        ], ratios=[0.5, 0.25, 0.25])
        return self.codingHeader

    def createCodingTable(self):
        codeViews = [
            ZStack([
                Rect(color=Color.steelBlue, keywords="rect", cornerRadius=10),
                Label(code.label, keywords="label")
            ], isDraggable=True, tag=code, keywords="codeStack", lockedWidth=200, lockedHeight=60, hideAllContainers=True)for code in self.codes
        ]
        shuffle(codeViews)

        self.codingTable = HStack([
            VStack(codeViews, name="question", containerArgs=[{"showEmpty": True}]),
            VStack([None] * len(self.codes), name="answer", containerArgs=[{"showEmpty": True, "tag": code.order} for code in self.codes])
        ], ratios=[0.3, 0.7])
        return self.codingTable

    def createCodingOptions(self):
        self.codingOptions = ZStack([
            self.createFileNameView(),
            HStack([
                None,
                self.createOpenSpreadsheetView(),
                self.createCodingFileView(),
                self.createOpenFilePath(),
                None
            ], hideAllContainers=True)
        ])
        return self.codingOptions

    def canDragView(self, view, container):
        return container.getParentStack().name == "question" or container.tag == view.tag.order

    def draggedView(self, view):

        stack = view.getParentStack()
        label = view.keyDown("label")
        success = stack.name == "answer"
        if stack.name == "answer":
            view.lock(lockedWidth=400)
            label.setFont(text=view.tag.line, fontSize=22)

            for container in stack.containers:
                success = success and container.view != None
        else:
            view.lock(lockedWidth=200)
            label.setFont(text=view.tag.label, fontSize=32)
        self.codingRunRect.color = Color.green if success else Color.gray
        self.codingRunButton.isDisabled = not success

    def createFileNameView(self):
        return Label("File: " + self.fileName, fontSize=18)

    def createOpenSpreadsheetView(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Excel")
        ], run=hp.openFile, tag=self.filePath + ".csv", lockedWidth=200)

    def createCodingFileView(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Code")
        ], run=hp.openFile, tag=self.codingFilePath, lockedWidth=200)

    def createOpenFilePath(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Select Data File")
        ], run=self.showFileExplorer, lockedWidth=200)

    def showFileExplorer(self, sender):
        view = ZStack([
            self.createFileExplorerView(),
            Button([
                Rect(Color.red, cornerRadius=10),
                Label("Close", fontSize=25)
            ], dy=1, lockedWidth=80, lockedHeight=60, offsetY=-50, run=self.hideFileExplorer)
        ])
        self.addView(view)
        self.updateAll()

    def createFileExplorerView(self):
        files = hp.getFiles(self.codingExamplePath, ".csv")
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

    def hideFileExplorer(self, sender):
        self.popView()
        self.updateAll()

    def runCodingTest(self, sender):
        self.codingAccLabel.setFont(text="Accuracy: {}%".format(round(100 * self.getTotalScore())))
        self.codingAccLabel.container.updateAll()


class InfoPage(ZStack):
    def __init__(self, files, **kwargs):
        self.files = files
        buttons = []
        for label, path in self.files:
            def setView(sender):
                labelView = sender.keyDown("label")
                if sender.isAlt():
                    labelView.setFont("Opening File...")
                else:
                    labelView.setFont("Click here to learn more about: " + labelView.name)

            buttons.append(Button(
                Label("", dx=-1, offsetX=10, keywords="label", name=label),
                tag=path,
                run=hp.openFile,
                setViewMethod=setView
            ))

        items = [VStack(
            [Label("More Information Below:", fontSize=48, dx=-1)] + buttons + [None] * 8, hideAllContainers=False
        )]
        super().__init__(items, **kwargs)


# =====================================================================
# Model Pages
# =====================================================================

# Decision Tree
class IntroDTPage(IntroPage):
    def __init__(self):
        description = ["    A tree has many analogies in real life and turns out that it has influenced",
                       "a wide area of machine learning, covering both classification and regression.",
                       "    Tree-based methods involve stratifying or segmenting the predictor space",
                       "into a number of simple regions. Since the set of splitting rules used to",
                       "segment the predictor space can be summarized in a tree, these types of",
                       "approaches are known as decision tree methods. The structure of a decision",
                       "tree includes:",
                       "       1) internal nodes corresponding to attributes (features)",
                       "       2) leaf nodes corresponding to the classification outcome",
                       "       3) edge denoting the assignment of the attribute."]
        super().__init__(description)


class ExampleDTPage(DTPage):
    def __init__(self, **kwargs):
        super().__init__(textboxScript=[
            ("Welcome to the Decision Tree Simulator!", 0, 0),
            (["To begin we will use a Decision Tree to",
              "analyze movie data. The objective of the",
              "model is to predict if movies would be",
              "liked or disliked based on type, length,",
              "and other characteristics."], 0.8, 0),
            (["Lets start by splitting the data shown on",
              "the right into separate groups of the",
              "same color"], -0.5, 0),
            ("Click on director to split the data into 3 groups", -0.8, 0),
            ("Next, click on director lass to subdivide the group", -0.8, 0),
            ("Finally, click on length to complete the tree", -0.8, 0),
            ("To show the full tree click on group name on the top", -0.8, 0),
            ("Congratulations on completing the tutorial", 0, 0)
        ], filePath="examples/decisionTree/movie", partition=False, textboxAudioPath="dt_final/dt")

        items = [
            HStack([
                VStack([
                    HStack([
                        self.createTableView(),
                        self.createTreeRoomView()

                    ], ratios=[0.6, 0.4]),
                    self.createHeaderButtons()
                ], ratios=[0.9, 0.1])
            ]),
            self.createNextTextbox()
        ]

        ZStack.__init__(self, items=items)


class ExceriseDTPage(DTPage):
    def __init__(self):
        super().__init__(filePath="examples/decisionTree/zoo")
        items = [
            HStack([
                VStack([
                    HStack([
                        self.createTreeListView(),
                        self.createTableView(),
                        self.createTreeRoomView()

                    ], ratios=[0.15, 0.55, 0.3]),
                    self.createHeaderButtons()

                ], ratios=[0.9, 0.1])
            ])
        ]
        # print(view)
        ZStack.__init__(self, items=items)


class CodingDTPage(CodingPage):
    def __init__(self, **kwargs):
        # Codes
        codes = [
            Code("model = DecisionTreeClassifier()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="Add Tree", filePath="examples/decisionTree/medical", codingFilePath="assets/treeExample.py",
                         codingIncMethod=self.incTrees, examplePath="examples/decisionTree", **kwargs)
        self.mainModel = DecisionTree(table=self.table)
        self.models.append(self.mainModel)

    def incTrees(self, sender):
        if len(self.models) >= 100:
            self.models = []
            self.codingAddLabel.setFont("Add Trees")

        else:
            for _ in range(10):
                train, test = self.trainTable.partition()
                self.mainModel = DecisionTree(table=train)
                self.models.append(self.mainModel)
            self.codingAddLabel.setFont("Trees: {}".format(len(self.models)))


class InfoDTPage(InfoPage):
    def __init__(self, **kwargs):
        files = [
            ("Generatation", "assets/decisiontree/GenerateDecisionTree.pdf"),
            ("Improvement", "assets/decisiontree/ImproveDecisionTree.pdf")
        ]
        super().__init__(files=files, **kwargs)


# KNN
class IntroKNNPage(ZStack):
    def __init__(self):
        items = [
            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),
            Label("""Welcome to the KNN Introduction Page""")
        ]
        super().__init__(items)


class ExampleKNNPage(KNNPage):
    def __init__(self):
        super().__init__(textboxScript=[
            ("Welcome to the KNN Simulator!", 0, 0)
        ], filePath="examples/linear/iris", partition=False)

        items = [
            self.createGraphView(clickPoint=self.clickPoint),
            self.createNextTextbox()
        ]
        ZStack.__init__(self, items=items)
        # print(view)

    def clickPoint(self, sender):
        dx, dy = hp.map(sender.lastClickX - self.modelView.x, 0.0, self.modelView.getWidth(), -1.0, 1.0), hp.map(sender.lastClickY - self.modelView.y, 0.0, self.modelView.getHeight(), -1.0, 1.0)
        view = Rect(color=self.classColors[self.mainModel.predictPoint(dx=dx, dy=dy)], dx=dx, dy=dy, lockedWidth=15, lockedHeight=15, tag=(dx, dy))
        self.modelMousePoints.append(view)
        self.modelView.addView(view)

    def hoverMouse(self, mouseX, mouseY):
        if self.hoverEnabled and self.isWithin(mouseX, mouseY):
            while self.modelView.peekView().name == "highlight":
                self.modelView.popView()
            x, y = hp.map(mouseX - self.modelView.x, 0.0, self.modelView.getWidth(), -1.0, 1.0), hp.map(mouseY - self.modelView.y, 0.0, self.modelView.getHeight(), -1.0, 1.0)
            for _, index, dx, dy in self.mainModel.getNeighbor(x, y):
                self.modelView.addView(
                    Ellipse(color=self.classColors[self.table.loc[index][self.table.targetName]], dx=dx, dy=dy, lockedWidth=20, lockedHeight=20, name="highlight")
                )
            self.updateAll()


class CodingKNNPage(KNNPage, CodingPage):
    def __init__(self, **kwargs):
        # Codes
        codes = [
            Code("model = KNeighborClassifier()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="K: 3", filePath="examples/linear/iris", codingFilePath="assets/treeExample.py", codingIncMethod=self.incMethod, examplePath="examples/knn", **kwargs)


class InfoKNNPage(InfoPage):
    def __init__(self, **kwargs):
        files = [
            ("Bayes and KNN", "assets/knn/TeachingMaterialsKNN.pdf"),
            ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)

# Linear


class IntroLinearPage(ZStack):
    def __init__(self):
        items = [
            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),
            Label("""Welcome to the Linear Regression Introduction Page""")
        ]
        super().__init__(items)


class ExampleLinearPage(LinearPage):
    def __init__(self):
        super().__init__(textboxScript=[
            ("Welcome to the Linear Regression Simulator!", 0, 0)
        ], filePath="examples/linear/iris", partition=False)

        items = [
            self.createGraphView(clickPoint=self.clickSlope, separateByTarget=False),
            self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]
        self.compModel = Linear(table=self.table, color=Color.blue)
        self.compTrainComplete = True

        self.modelLine, self.modelError, self.modelEq = self.createLines(color=self.mainModel.color, errorOffset=-120, eqOffset=-90, dx=-1, dy=1)
        self.compLine, self.compError, self.compEq = self.createLines(color=self.compModel.color, errorOffset=90, eqOffset=120, dx=1, dy=-1, offsetX=-110)

        ZStack.__init__(self, items=items)
        # print(view)

    def update(self):
        # print("UPDATE")
        if not self.compTrainComplete:
            self.compModel.setSize(self.modelView)
            self.compTrainComplete = not self.compModel.fit()
            self.compLine.points = self.compModel.getEdgePoints()
            self.compError.setFont(text="ML Error: {}".format(round(self.compModel.getError(), 4)))
            self.compEq.setFont(text=self.compModel.getEqString())

            # print("COMP:", self.compModel.cef, "ERROR:", self.compModel.dJ)


class QuadLinearPage(LinearPage):
    def __init__(self):
        super().__init__(textboxScript=[
            ("Welcome to the Linear Regression Simulator!", 0, 0)
        ], filePath="examples/linear/iris", partition=False)
        self.mainModel.n = 2

        items = [
            self.createGraphView(clickPoint=self.clickSlope, separateByTarget=False),
            # self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]

        self.modelLine, self.modelError, self.modelEq = self.createLines(color=self.mainModel.color, errorOffset=-120, eqOffset=-90, dx=-1, dy=1)

        ZStack.__init__(self, items=items)
        # print(view)

    def update(self):
        pass


class CodingLinearPage(CodingPage):
    def __init__(self, **kwargs):
        # Codes
        codes = [
            Code("model = LinearRegression()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="--", filePath="examples/linear/iris",
                         codingFilePath="assets/treeExample.py", codingIncMethod=self.incMethod, examplePath="examples/linear", **kwargs)
        self.mainModel = Linear(table=self.table)
        self.models.append(self.mainModel)

    def runCodingTest(self, sender):
        self.codingAccLabel.setFont(text="Error: {}".format(self.mainModel.getModelError(self.finalTestTable)))
        self.codingAccLabel.container.updateAll()


class InfoLinearPage(InfoPage):
    def __init__(self, **kwargs):
        files = [
            ("Linear Regression", "assets/linear/LinearRegression.pdf"),
            ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)


# Logistic
class IntroLogisticPage(ZStack):
    def __init__(self):
        items = [

            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),
            Label("""Welcome to the Logistic Regression Introduction Page""")

        ]
        super().__init__(items)


class ExampleLogisticPage(LogisticPage):
    def __init__(self):
        super().__init__(textboxScript=[
            ("Welcome to the Logistic Regression Simulator!", 0, 0)
        ], filePath="examples/logistic/sigmoid", partition=False)

        items = [
            self.createGraphView(clickPoint=self.clickSlope, separateByTarget=False),
            # self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]
        self.modelLine, self.modelError, self.modelEq = self.createLines(color=self.mainModel.color, errorOffset=-120, eqOffset=-90, dx=-1, dy=1)

        ZStack.__init__(self, items=items)
        # print(view)

    def update(self):
        # print("UPDATE")
        pass
        # if not self.compTrainComplete:
        #     self.compModel.setSize(self.modelView)
        #     self.compTrainComplete = not self.compModel.fit()
        #     self.compLine.points = self.compModel.getEdgePoints()
        #     self.compError.setFont(text="ML Error: {}".format(round(self.compModel.getError(), 4)))
        #     self.compEq.setFont(text=self.compModel.getEqString())

        # print("COMP:", self.compModel.cef, "ERROR:", self.compModel.dJ)


class CodingLogisticPage(CodingPage):
    def __init__(self, **kwargs):
        # Codes
        codes = [
            Code("model = LogisticRegression()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="--", filePath="examples/linear/iris",
                         codingFilePath="assets/treeExample.py", codingIncMethod=self.incMethod, examplePath="examples/linear", **kwargs)
        self.mainModel = Linear(table=self.table)
        self.models.append(self.mainModel)

    def runCodingTest(self, sender):
        self.codingAccLabel.setFont(text="Error: {}".format(self.mainModel.getModelError(self.finalTestTable)))
        self.codingAccLabel.container.updateAll()


class InfoLogisticPage(InfoPage):
    def __init__(self, **kwargs):
        files = [
            # ("Linear Regression", "assets/linear/LinearRegression.pdf"),
            # ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)


# =====================================================================
# Support Classes
# =====================================================================


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
