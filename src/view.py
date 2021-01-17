from graphics import ZStack, VStack, Label, Rect, HStack, Grid, Button, Color, Points, Container, Image
import helper as hp
# from table import *
# from models import *
# from random import shuffle
# from math import sin, cos, pi
# from elements import *
# from comp import *
# import statistics as stat
# from time import time
from base import SingleModel, MultiModel

# Solo Views
modelTitle = "Temp"

# =====================================================================
# Non-model based views
# =====================================================================


class MouseDebug(ZStack):
    def __init__(self):
        super().__init__([
            Rect(color=Color.white, keywords="mRect", border=0),
            Label(text="", fontSize=15, color=Color.black, keywords="text")
        ], lockedWidth=80, lockedHeight=20, dx=-1, dy=1)


class TextboxView(ZStack):

    def __init__(self, textboxScript=None, textboxAudioPath=None, **kwargs):
        text, dx, dy = textboxScript[0]
        soundName = textboxAudioPath + str(1) if textboxAudioPath != None else None
        label = Label(text=text, color=Color.black, fontSize=15)
        audio = Button(Image(imageName="audio.png", lockedWidth=50, lockedHeight=50), soundName=soundName, dx=1, dy=1, offsetX=-20, offsetY=-20, lockedWidth=60, lockedHeight=60)

        # super().__init__(items=None)
        super().__init__([
            Button([
                Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
                label
            ], run=self.pressTextbox),
            audio
        ], dx=dx, dy=dy, lockedWidth=450, lockedHeight=200, hideAllContainers=True, **kwargs)

        self.textboxScript = textboxScript
        self.textboxAudioPath = textboxAudioPath
        self.textboxIndex = 1000  # 0
        self.label = label
        self.audio = audio

    def pressTextbox(self, sender):
        self.textboxIndex += 1
        self.updateTextbox()

    def updateTextbox(self):
        if self.textboxIndex >= len(self.textboxScript):
            self.delink(allowButtonUpdate=False)
            return

        text, dx, dy = self.textboxScript[self.textboxIndex]
        soundName = self.textboxAudioPath + str(self.textboxIndex + 1) if self.textboxAudioPath != None and self.textboxIndex != None else None

        self.label.setFont(text=text)
        self.setAlignment(dx=dx, dy=dy)
        self.audio.setSoundName(soundName=soundName)
        self.updateAll()


class IntroView(ZStack):

    def __init__(self, label, **kwargs):
        items = [
            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),
            label
        ]
        super().__init__(items=items, **kwargs)


class InfoView(VStack):
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

        items = [Label("More Information Below:", fontSize=48, dx=-1)] + buttons + [None] * 8
        super().__init__(items, hideAllContainers=False, **kwargs)

# =====================================================================
# Model based views
# =====================================================================


class SingleModelView(SingleModel):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, table=model.table, **kwargs)
        self.table.addGraphic(self)


class MultiModelView(MultiModel):
    def __init__(self, models=[], compModels=[], **kwargs):
        super().__init__(table=models[0].table if models else (compModels[0].table if compModels else None), **kwargs)
        self.table.addGraphic(self)
        for model in models:
            self.addModel(model)
        for model in compModels:
            self.addCompModel(model)


class TableView(SingleModelView, Grid):

    def __init__(self, **kwargs):
        SingleModelView.__init__(self, **kwargs)
        Grid.__init__(self, table=self.table, createCellViewMethod=self.createCellView, **kwargs)

    # def updateTableView(self, **kwargs):
    #     prevContainer = self.tableView.container if self.tableView != None else None
    #     self.createTableView()
    #     if prevContainer != None:
    #         self.tableView.setContainer(prevContainer)

    # need ref to grid since it is not created yet

    def createCellView(self, grid, index):
        if index == None:
            return None
        colIndex = index % grid.cols
        table = grid.model.getTable()
        # print(grid.items[index], type(grid.items[index]))
        displayItem = str(int(grid.items[index]) == 1) if index >= grid.cols and table.isBoolCol[colIndex] else table.map(colIndex, grid.items[index])  # mapped and str

        # Rect Colors
        # Header: Steel Blue (index < grid.cols)
        # Label: Class Color (colIndex == 0)
        # Other: Light Blue
        return ZStack([
            Rect(color=Color.steelBlue if index < grid.cols else(table.classColors[str(grid.items[index])]
                                                                 if colIndex == 0 else Color.lightSteelBlue), border=3, cornerRadius=5, keywords="rect"),
            Label(text=displayItem, fontSize=15, color=Color.white, keywords="label")
        ])

    def shiftTable(self, dy):
        self.shift(dy=dy)
        self.table.tableChange(self.table.selectedCol, isSelect=True)
        # self.colorTableTargets()
        # self.updateHeaderSelectionButtons()
        self.updateAll()

    def scrollUp(self):
        self.shiftTable(dy=1)

    def scrollDown(self):
        self.shiftTable(dy=-1)

    def tableChange(self, colIndex, isSelect, isLock, isNewTable):
        # creates infinite loop...
        if isNewTable:
            self.build(table=self.model.getTable())
            self.updateAll()
            # TableView(model=self.model).setContainer(self.container)
            return
        if colIndex == None or isSelect == None:
            return
        for index in range(colIndex + self.cols, self.length, self.cols):
            view = self.getView(index)
            if view != None:
                view.keyDown("rect").color = self.table.classColors[self.items[index]] if colIndex == 0 else (Color.steelBlue if isSelect else Color.lightSteelBlue)


class HeaderButtons(SingleModelView, HStack):

    def __init__(self, **kwargs):
        SingleModelView.__init__(self, **kwargs)
        HStack.__init__(self, [
            Button([
                Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
                Label(self.table.xNames[i], fontSize=25, color=Color.white)
            ], isOn=False, tag=i + 1, run=self.pressButton) for i in range(self.table.colCount)
        ], **kwargs)

    def pressButton(self, sender):
        self.table.tableChange(self.table.selectedCol, isSelect=False)
        self.table.tableChange(sender.tag, isSelect=sender.isOn)

    def tableChange(self, column, isSelect, isLock, isNewTable):
        if column == None:
            return
        # print("Update Button:", column, "Is Select:", isSelect, "Is Lock:", isLock)

        button = self.getView(column - 1)
        rect = button.getView(0)
        if isLock:
            rect.strokeColor = Color.gray
            button.isDisabled = True
            button.setOn(isOn=False)
        else:
            button.isDisabled = False
            if isSelect:
                rect.strokeColor = Color.yellow
                button.setOn(isOn=True)
            else:
                rect.strokeColor = Color.darkGray
                button.setOn(isOn=False)

        # def updateButtons(self):
        #     for button in self.getViews():
        #         rect = button.getView(0)
        #         column = self.table.xNames[button.tag - 1]

        #         if self.model.isLockedColumn(column):
        #             rect.strokeColor = Color.gray
        #             button.isDisabled = True
        #             button.setOn(isOn=False)
        #         else:
        #             button.isDisabled = False
        #             if self.table.selectedCol == button.tag:
        #                 rect.strokeColor = Color.yellow
        #                 button.setOn(isOn=True)
        #             else:
        #                 rect.strokeColor = Color.darkGray
        #                 button.setOn(isOn=False)


class TreeRoom(SingleModelView, VStack):
    def __init__(self, **kwargs):
        SingleModelView.__init__(self, **kwargs)
        view = Points(pts=self.model.getCircleLabelPts())
        self.backButton = Button([
            Rect(color=Color.gray, isDisabled=True),
            Label(text="Back")  # , isDisabled=True
        ], run=self.goBack, setViewMethod=self.updateBackButton)

        VStack.__init__(self, items=[
            view,
            self.backButton
        ], ratios=[0.9, 0.1])
        self.modelBranch = Branch(view=view, disjoint=Container())

    # def updateRoom(self):
    #     self.setPts(pts=self.model.getCircleLabelPts())
    #     self.modelBranch = Branch(view=self, disjoint=Container())

    def updateBackButton(self, button):
        button.isDisabled, button.getView(0).color = (True, Color.gray) if self.model.isRoot() else (False, Color.blue)

    def tableChange(self, colIndex, isSelect, isLock, isNewTable):
        self.updateBackButton(self.backButton)
        if isSelect == None or isNewTable:
            return
        # rect = sender.getCousin(0)
        if isSelect:  # True
            self.splitTree(colIndex - 1)
            # rect.strokeColor = Color.white
        else:  # False
            self.removeNodeTree()
            # rect.strokeColor = Color.gray
        # self.updateBackButton(self.backButton)
        self.modelBranch.getContainer().updateAll()

    def splitTree(self, colIndex):
        self.model.add(colIndex)
        container = self.modelBranch.getContainer()
        prevStack = type(self.modelBranch.disjoint.keyUp("div"))

        stack = VStack if prevStack != VStack else HStack
        label = [
            Button(
                Label(text="{}:{}".format(self.model.getParentColName(), self.model.getValue()), fontSize=10,
                      color=Color.white, dx=-0.95, dy=-1),
                run=self.goBack)
        ] if self.model.getParent() != None else []

        treeChildren = self.model.getChildren()
        totalClassCount = len(treeChildren)
        for child in treeChildren:
            totalClassCount += child.table.classCount
        # print("Total Class Count:", totalClassCount)

        self.modelBranch.setView(view=ZStack(items=label + [
            stack(items=[
                VStack([
                    Button(Label(text="{}:{}".format(self.model.getColName(), self.model.getChild(i).value), fontSize=10, dx=-1), run=self.goBack),
                    Button(Points(pts=self.model.getCircleLabelPts(self.model.getChild(i).table), isConnected=False), run=self.goForward, tag=i)
                ], ratios=[0.05, 0.95], keywords="dotStack") for i in range(len(treeChildren))
            ], ratios=[
                (child.table.classCount + 1) / totalClassCount for child in treeChildren
            ], border=20 if self.model.getParent() != None else 0, keywords="div")
        ], keywords=["z"]))

    def goForward(self, sender):
        if self.model.hasChildren():
            # print("Forward: Children | Selected Col:", self.table.selectedCol)
            self.goToChildTree(index=sender.tag)
            # print("Forwar2: Children | Selected Col:", self.table.selectedCol)

            self.table.tableChange(self.table.selectedCol, isLock=True, isSelect=False, isNewTable=True)
            self.modelBranch.getContainer().updateAll()
            # self.updateContainers()

    def goBack(self, sender):
        if not self.model.isRoot():
            self.goBackTree()
            self.table.tableChange(self.table.selectedCol, isSelect=False, isNewTable=True)
            self.table.tableChange(None, isLock=False, isSelect=True, isNewTable=True)

    def removeNodeTree(self):
        self.model.remove()
        self.modelBranch.setView(view=Points(pts=self.model.getCircleLabelPts()))

    def goBackTree(self):
        self.model.goBack()
        self.modelBranch.move(self.modelBranch.disjoint.keyUp("z"))
        self.modelBranch.getContainer().updateAll()

    def goToChildTree(self, index):
        self.model.go(index=index)
        container = self.modelBranch.view.keyDown("div")[index]
        stack = container.keyDown("z")
        moveView = stack if stack != None else container.keyDown("dotStack")
        # print("MOVE:", moveView)
        self.modelBranch.move(moveView)

    # def selectColumn(self, sender):
    #     super().selectColumn(sender)
    #     rect = sender.getCousin(0)
    #     if sender.isOn:
    #         self.splitTree(column=self.model.getColName(sender.tag))
    #         rect.strokeColor = Color.white
    #     else:
    #         self.model.remove()
    #         rect.strokeColor = Color.gray
    #     self.modelBranch.getContainer().updateAll()


class TreeList(SingleModelView, VStack):
    def __init__(self, **kwargs):
        SingleModelView.__init__(self, **kwargs)

        self.totalScoreLabel = Label("Total " + self.model.defaultScoreString(), fontSize=20)
        self.totalScore = ZStack([
            Rect(color=Color.steelBlue),
            self.totalScoreLabel
        ], lockedHeight=50, dy=1)

        VStack.__init__(self, [None] * 9 + [
            self.totalScore,
            Button([
                Rect(color=Color.steelBlue, cornerRadius=10),
                Label("Save Tree")
            ], run=self.saveTree, lockedHeight=50, dy=1)
        ])

    def saveTree(self, sender):
        bagItem = ZStack([
            Rect(color=Color.steelBlue),
            Label(text=self.model.curr.getScoreString(), fontSize=20)
        ], dy=1)
        bagItem.setContainer(self[len(self.model)])

        self.totalScoreLabel.setFont(text="Total " + self.model.getScoreString())
        self.model.saveTree()
        self.table.tableChange()
        self.updateAll()
        # self.table, self.localTestTable = self.trainTable.partition()

        # self.model = DecisionTree(table=self.table)
        # self.models.append(self.model)
        # self.createTreeRoomView()
        # self.updateTableView()

        # self.bag.updateAll()
        # self.modelView.container.updateAll()
        # self.tableView.container.updateAll()
        # self.updateHeaderSelectionButtons()

    def tableChange(self, colIndex, isSelect, isLock, isNewTable):
        pass


class GraphView(MultiModelView, ZStack):
    def __init__(self, hasAxis=False, enableUserPts=False, **kwargs):
        self.graphics = []
        MultiModelView.__init__(self, **kwargs)
        self.hasAxis = hasAxis
        self.userPts = Points(maxPts=100, isCircle=False, ptSize=8) if enableUserPts else None
        self.legend = None

        self.graphGrid = Button([self.userPts] + [self.legend] + self.graphics, run=self.clickGraph)

        ZStack.__init__(self, [
            HStack([
                VStack([
                    self.createAxis(model=self.models[0], isVertical=True),
                    None,
                ], ratios=[0.9, 0.1]),
                VStack([
                    self.graphGrid,
                    self.createAxis(model=self.models[0], isVertical=False),
                ], ratios=[0.9, 0.1])
            ], ratios=[0.08, 0.92]),
            self.createAddCompButton() if self.compModels else None
        ], **kwargs)
        # if self.drawModel:f
        #     self.modelLine, self.modelError, self.modelEq = self.createLines(color=self.model.color, errorOffset=-120, eqOffset=-90, dx=-1, dy=1)
        # if self.drawComp:
        #     self.compLine, self.compError, self.compEq = self.createLines(color=self.compModel.color, errorOffset=90, eqOffset=120, dx=1, dy=-1, offsetX=-150)

    def addModel(self, model):
        super().addModel(model)
        # create model equation label & error label
        model.addGraphics(("pts", Points(pts=[], color=model.color, isConnected=True)))
        if model.isRegression:
            model.addGraphics(
                ("err", Label(model.defaultScoreString(), fontSize=15, color=model.color, dx=1, dy=-1, offsetY=100 + 45 * len(self.models), offsetX=-15)),
                ("eq", Label("Y=--", fontSize=15, color=model.color, dx=1, dy=-1, offsetY=120 + 45 * len(self.models), offsetX=-15))
            )
        if model.isClassification:
            model.addGraphics(
                ("acc", Label(model.getScoreString(), fontSize=15, color=model.color, dx=1, dy=-1, offsetY=100 + 45 * len(self.models), offsetX=-15))
            )
        self.graphics += [graphic for graphic in model.graphics]

    def addCompModel(self, model):
        super().addCompModel(model)
        self.addModel(model)

    def createLegendView(self, model):
        if not model.isCategorical:
            self.legend = None
            return

        legendItems = [Label("Legend", fontSize=30, color=Color.white)] + [
            Label(str(label), fontSize=30, color=self.model.classColors[label]) for label in self.table.classSet
        ]

        self.legend = ZStack([
            Rect(color=Color.backgroundColor, strokeWidth=3, strokeColor=Color.steelBlue, cornerRadius=10),
            VStack(legendItems, ratios=[0.8 / len(legendItems) for item in legendItems], offsetY=10, hideAllContainers=True)
        ], lockedWidth=150, lockedHeight=180, dx=0.8, dy=-0.8, hideAllContainers=True)

    # def createDots(self):
    #     items = []
    #     for index, row in self.table.iterrows():
    #         items.append(Ellipse(color=self.model.classColors[row[self.table.targetName]] if self.model.isCategorical else Color.steelBlue,
    #                              strokeColor=Color.white, strokeWidth=3, dx=row[self.table.first()], dy=row[self.table.second()], lockedWidth=15, lockedHeight=15))
    #     return items

    def createAxis(self, model, isVertical):
        return None if not self.hasAxis else ZStack([
            Rect(color=Color.steelBlue, strokeColor=Color.darkGray, strokeWidth=4, cornerRadius=10),
            Label(model.colNameB if isVertical else model.colNameA, fontSize=25, color=Color.white, isVertical=isVertical)
        ])

    # def createLines(self, color, errorOffset, eqOffset, **kwargs):
    #     lines = Lines(color=color)
    #     errorLabel = Label("Error: --", color=color, offsetY=errorOffset, **kwargs)
    #     eqLabel = Label("Y=--", color=color, offsetY=eqOffset, **kwargs)
    #     self.modelView.addAllViews(lines, errorLabel, eqLabel)
    #     return lines, errorLabel, eqLabel

    def clickGraph(self, sender):
        if self.userPts != None:
            dx = hp.map(sender.lastClickX, sender.x, sender.x + sender.getWidth(), -1, 1)
            dy = hp.map(sender.lastClickY, sender.y, sender.y + sender.getHeight(), -1, 1)
            self.userPts.addPt((dx, dy, self.models[0].color))

        for model in self.models:
            if model.isUserSet:
                x = hp.map(sender.lastClickX, sender.x, sender.x + sender.getWidth(), model.minX1, model.maxX1)
                y = hp.map(sender.lastClickY, sender.y, sender.y + sender.getHeight(), model.minX2, model.maxX2)
                model.addPt(x, y)
        # if self.drawModel:
        #     points = self.model.addPt((sender.lastClickX, sender.lastClickY))

        #     if points != None:
        #         self.modelLine.points = points
        #         self.modelError.setFont(text="Error: {}".format(round(self.model.getError(), 4)))
        #         self.modelEq.setFont(text=self.model.getEqString())
        #     else:
        #         self.modelLine.points = []

    def createIncButton(self, **kwargs):
        pass

    def incMethod(self, sender):
        pass

    def createAddCompButton(self):
        self.addCompButton = Button([
            Rect(color=Color.backgroundColor, cornerRadius=10, strokeColor=Color.steelBlue, strokeWidth=3),
            Label("ML Results", fontSize=15)
        ], lockedWidth=150, lockedHeight=80, dx=1, dy=-1, offsetX=-10, offsetY=10, run=self.startComp)
        return self.addCompButton

    def startComp(self, sender):
        for model in self.compModels:
            model.startTraining()

    def update(self):
        for model in self.compModels:
            if model.isRunning:
                model.fit()

                # self.compLine.points = self.compModel.getEdgePoints()
                # self.compError.setFont(text="ML Error: {}".format(round(self.compModel.getError(), 4)))
                # self.compEq.setFont(text=self.compModel.getEqString())

            # print("COMP:", self.compModel.cef, "ERROR:", self.compModel.dJ)

    def hoverMouse(self, x, y):
        if self.hoverEnabled:
            for model in self.models:
                if model.isUserSet:
                    x1 = hp.map(x, self.graphGrid.x, self.graphGrid.x + self.graphGrid.getWidth(), model.minX1, model.maxX1)
                    y1 = hp.map(y, self.graphGrid.y, self.graphGrid.y + self.graphGrid.getHeight(), model.minX2, model.maxX2)
                    # print("H:", x1, y1, "X1:", model.minX1, model.maxX1, "X2:", model.minX2, model.maxX2)
                    model.addPt(x1, y1, storePt=False)
            # points = self.model.addPt((x, y), storePoint=False)
            # if points != None:
            #     self.modelLine.points = points
            #     self.modelError.setFont(text="Error: {}".format(round(self.model.getError(), 4)))
            #     self.modelEq.setFont(text=self.model.getEqString())


class KNNGraphView(GraphView):

    def clickGraph(self, sender):
        super().clickGraph(sender)
        if self.userPts != None:
            x = [
                hp.map(sender.lastClickX, sender.x, sender.x + sender.getWidth(), self.models[0].minX1, self.models[0].maxX1),
                hp.map(sender.lastClickY, sender.y, sender.y + sender.getHeight(), self.models[0].minX2, self.models[0].maxX2),
            ]
            pred = self.models[0].predict(x)

            # print("CLICK:", pred, self.models[0].table.classColors[pred])

            # change color of most recent one
            self.userPts.setColor(-1, self.models[0].table.classColors[str(pred)])

    def incMethod(self, sender):
        self.model.k += 2
        if self.model.k > 10:
            self.model.k = 1
        self.codingAddLabel.setFont("K: {}".format(self.model.k))

    def createIncButton(self, **kwargs):
        self.codingAddLabel = Label("K: {}".format(self.model.k))
        return Button([
            Rect(color=Color.backgroundColor, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
            self.codingAddLabel
        ], run=self.incMethod, lockedWidth=130, lockedHeight=80, **kwargs)


class LinearGraphView(GraphView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SVMGraphView(GraphView):
    def addModel(self, model):
        model.addGraphics(("pts2", Points(pts=[], color=Color.gray, isConnected=True)))
        model.addGraphics(("pts3", Points(pts=[], color=Color.gray, isConnected=True)))
        super().addModel(model)

# =====================================================================
# Support classes
# =====================================================================


class Branch:
    """
                              () -----Prev View
    Display Container- ()     ||
                       ||     () -----Disjoint Container
    View-------------- () _ _ /
                       ||
    Next View--------- ()
    """

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
            nextView.setContainer(container=self.view.container)  # disconnect nextView from prev container # connect new view
            self.view.setContainer(container=self.disjoint)  # set prev view to disjoint
            self.disjoint = nextDisjoint  # move prev view to disjoint
            self.view = nextView  # move ptr to new view
