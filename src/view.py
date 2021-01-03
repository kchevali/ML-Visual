from graphics import *
import helper as hp
import viewHelper as vp
from table import *
from models import *
from random import shuffle
from math import sin, cos, pi
from elements import *
from comp import *
import statistics as stat
from time import time

# Solo Views


# NO TOUCH unless broken
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

        super().__init__([
            Button([
                Rect(color=Color.white, strokeColor=Color.steelBlue, strokeWidth=3, cornerRadius=10),
                label
            ], run=self.pressTextbox),
            audio
        ], dx=dx, dy=dy, lockedWidth=450, lockedHeight=200, hideAllContainers=True, **kwargs)

        self.textboxScript = textboxScript
        self.textboxAudioPath = textboxAudioPath
        self.textboxIndex = 0
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

# Need to Work On!


class ModelView(ZStack):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model


class GroupView:
    # def __init__(self,views=[], **kwargs):
    #     super().__init__(items=views, **kwargs)
    #     self.classType = classType
    #     self.classArgs = classArgs
    #     self.models = models
    #     # self.addAllModelViews()#should be called after __init__() is done
    #     # it is wrong for the parent to assume the subclasses are init when the parent is done

    def createViews(self, models):
        pass

    def addModelViews(self):
        for modelView in self.getViews():
            modelView.addView(modelView.createModelView()())

    def rebuild(self):
        for modelView in self.getViews():
            modelView.clear()
        self.addAllModelViews()


class TableView(BaseView):

    def createModelView(self, model):
        return Grid(model=model, createCellViewMethod=self.createCellView)

    # def updateTableView(self, **kwargs):
    #     prevContainer = self.tableView.container if self.tableView != None else None
    #     self.createTableView()
    #     if prevContainer != None:
    #         self.tableView.setContainer(prevContainer)

    # need ref to tableview since it is not created yet
    def createCellView(self, grid, index):
        if index == None:
            return None
        colIndex = index % grid.cols
        # self.model is the current model
        return ZStack([
            Rect(color=Color.steelBlue if index < grid.cols else(grid.model.table.classColors[grid.items[index]]
                                                                 if colIndex == 0 else Color.lightSteelBlue), border=3, cornerRadius=5, keywords="rect"),
            Label(text=str(grid.items[index]), fontSize=20, color=Color.white, keywords="label")
        ])

    def updateTableCol(self, colIndex):
        if colIndex == None:
            return
        isSelected = self.selectedCol == colIndex
        for index in range(colIndex + self.cols, self.length, self.cols):
            view = self.getView(index)
        if view != None:
            view.keyDown("rect").color = self.model.table.classColors[self.items[index]] if colIndex == 0 else (Color.steelBlue if isSelected else Color.lightSteelBlue)

    def shiftTable(self, dy):
        for model in self.models:
            self.shift(dy=dy)
            self.colorTableTargets()
            self.updateHeaderSelectionButtons()
            self.updateAll()

    def scrollUp(self):
        self.shiftTable(dy=1)

    def scrollDown(self):
        self.shiftTable(dy=-1)


class HeaderView(BaseView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selectedCol = None

    def createModelView(self, model):
        return HStack([
            Button([
                Rect(color=Color.steelBlue, strokeColor=Color.white, strokeWidth=4, cornerRadius=10),
                Label(model.table.xNames[i], fontSize=25, color=Color.white)
            ], isOn=False, tag=(model, i + 1), run=self.selectColumn) for i in range(model.table.colCount)
        ])

    def selectColumn(self, sender):
        model, colIndex = sender.tag

        self.selectedCol, prevCol = colIndex if sender.isOn else None, self.selectedCol

        # terminate if table doesn't exist or no change is needed
        if self.selectedCol == prevCol:
            return
        for buttons in self.getViews():
            self.updateButtons(buttons=buttons, model=model)
            # self.updateTableCol(buttons=buttons, prevCol)
            # self.updateTableCol(buttons=buttons, self.selectedCol)

    def updateButtons(self, buttons, model):
        for view in buttons.getViews():
            rect = view.getView(0)
            button = view.getView(1)
            column = model.table.xNames[button.tag - 1]

            if self.model.isLockedColumn(column):
                rect.strokeColor = Color.gray
                button.isDisabled = True
                button.setOn(isOn=False)
            else:
                button.isDisabled = False
                if self.selectedCol == button.tag:
                    rect.strokeColor = Color.yellow
                    button.setOn(isOn=True)
                else:
                    rect.strokeColor = Color.darkGray
                    button.setOn(isOn=False)


class GraphView(BaseView):
    def __init__(self, hasAxis=False, enableUserPts=False, **kwargs):
        super().__init__(**kwargs)
        self.hasAxis = hasAxis
        self.userPts = Points(maxPts=100) if enableUserPts else None
        self.compModels = []
        self.legend = None

    def createModelView(self, model):
            # self.modelView = ZStack([
            #     Button(self.createDots(), limit=200, run=self.clickGraph),
            #     self.legend,
            #     self.createIncButton(dx=-1, dy=1)
            # ], limit=100)
        self.graphGrid = Button([self.userPts] + [self.legend] + model.graphics, run=self.clickGraph)
        self.graphView = HStack([
            VStack([
                self.createAxis(model=self.models[0], isVertical=True),
                None,
            ], ratios=[0.9, 0.1]),
            VStack([
                self.graphGrid,
                self.createAxis(model=self.models[0], isVertical=False),
            ], ratios=[0.9, 0.1])
        ], ratios=[0.08, 0.92])
        return self.graphView
        # if self.drawModel:f
        #     self.modelLine, self.modelError, self.modelEq = self.createLines(color=self.model.color, errorOffset=-120, eqOffset=-90, dx=-1, dy=1)
        # if self.drawComp:
        #     self.compLine, self.compError, self.compEq = self.createLines(color=self.compModel.color, errorOffset=90, eqOffset=120, dx=1, dy=-1, offsetX=-150)

    def addModel(self, model):
        # create model equation label & error label
        model.addGraphics(
            ("pts", Points(pts=[], color=model.color, isConnected=True)),
            ("err", Label("Error: --", fontSize=15, color=model.color, dx=1, dy=-1, offsetY=100 + 45 * len(self.models), offsetX=-15)),
            ("eq", Label("Y=--", fontSize=15, color=model.color, dx=1, dy=-1, offsetY=120 + 45 * len(self.models), offsetX=-15))
        )

        # add model to list
        self.models.append(model)

    def addCompModel(self, model):
        self.addModel(model)
        self.compModels.append(model)

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
        ], lockedWidth=150, lockedHeight=80, dx=1, dy=-1, offsetX=-10, offsetY=10, tag=0, run=self.startComp)
        return self.addCompButton

    def startComp(self, sender):
        if(sender.tag < len(self.compModels)):
            self.compModels[sender.tag].isRunning = True

    def update(self):
        # print("UPDATE")
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


class CodingView(ZStack):
    def __init__(self, **kwargs):
        super().__init__([
            VStack([
                self.createCodingHeader(),
                self.createCodingTable(),
                self.createCodingOptions(table=table)
            ], ratios=[0.1, 0.8, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the Coding Tutorial!", 0, 0),
                (["On this page, we will show the basics on how to",
                    "run a " + modelTitle + " on Python"], 0, 0),
                (["Lets begin by dragging the code labels on the",
                    "left column to the correct spots on right"], 0.5, 0),
                ("Can you figure out the correct order?", 0, 0),
                (["Once you successfully set the code blocks,",
                    "add some trees and run your code!"], 0, 0)
            ])
        ], **kwargs)

    def createCodingHeader(self):
        self.codingRunRect = Rect(color=Color.gray, cornerRadius=10)
        self.codingRunButton = Button([
            self.codingRunRect,
            Label("Run")
        ], hideAllContainers=True, lockedWidth=240, run=self.runCodingTest, isDisabled=True)

        self.codingScoreLabel = Label(self.model.defaultScoreString())
        self.codingAddLabel = Label(self.codingAddString)
        self.codingHeader = HStack([
            self.codingRunButton,
            self.codingScoreLabel,
            Button([
                Rect(color=Color.steelBlue, cornerRadius=10),
                self.codingAddLabel
            ], run=self.incMethod)
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

    def createCodingOptions(self, table):
        self.codingOptions = ZStack([
            self.createFileNameView(table=table),
            HStack([
                None,
                self.createOpenSpreadsheetView(table=table),
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

    def createFileNameView(self, table):
        return Label("File: " + table.fileName, fontSize=18)

    def createOpenSpreadsheetView(self, table):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Excel")
        ], run=hp.openFile, tag=table.filePath + ".csv", lockedWidth=200)

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
        length = 10
        self.fileExplorer = ZStack([
            Rect(Color.backgroundColor, border=0),
            VStack([
                ZStack([
                    Rect(color=Color.steelBlue, cornerRadius=10),
                    Label("Files", fontSize=35)
                ])] + [
                Button([
                    Rect(color=Color.steelBlue, cornerRadius=10),
                    Label(fileName.split(".")[0], fontSize=25)
                ], name=fileName, lockedWidth=150) for fileName in files
            ] + [None] * (10 - len(files) - 1), ratios=[1.0 / length] * length)
        ], lockedWidth=350, lockedHeight=600)
        return self.fileExplorer

    def hideFileExplorer(self, sender):
        self.popView()
        self.updateAll()

    def runCodingTest(self, sender):
        self.codingScoreLabel.setFont(text=getScoreString())
        self.codingScoreLabel.container.updateAll()
