from graphics import Button, Label, HStack, Color, ZStack, Rect, VStack
import helper as hp
from table import Table
from models import DecisionTree, RandomForest, KNN, Linear, Logistic, SVM
from random import shuffle
from elements import createLabel, createButton
from comp import Data
import statistics as stat
from time import time
from view import TableView, TreeRoom, HeaderButtons, TreeList, GraphView, KNNGraphView, SVMGraphView, LinearGraphView, TextboxView, IntroView, InfoView
from base import SingleModel, MultiModel
import numpy as np


modelTitle = ""


# =====================================================================
# Screen Objects
# =====================================================================

# =====================================================================
# START UP CLASSES
# =====================================================================


class ModelPage(VStack):
    def __init__(self, content, title, pages=[], includeTaskList=True):

        self.content = content
        self.title = title
        self.pages = pages
        self.includeTaskList = includeTaskList
        self.taskListLength = 8

        self.content.modelPage = self

        items = [
            createLabel(self.title, views=[Button(Label("<"), run=self.replaceSelf, tag=MenuPage, lockedWidth=40, lockedHeight=40, dx=-1, offsetX=20) if type(self) != MenuPage else None]),
            HStack([
                self.createTaskList(),
                self.content
            ], ratios=[0.15, 0.85]) if self.includeTaskList else self.content

        ]
        super().__init__(items, ratios=[0.08, 0.92])

    def createTaskList(self):
        return VStack([
            createButton(text=task, color=Color.orange, tag=page, run=self.replaceContent) for task, page in self.pages
        ] + [None] * (self.taskListLength - len(self.pages)), ratios=[1.0 / self.taskListLength] * self.taskListLength)

    def draggedView(self, view):
        return self.content.draggedView(view=view)

    def canDragView(self, view, container):
        return self.content.canDragView(view, container)

    def scrollUp(self):
        self.content.scrollUp()

    def scrollDown(self):
        self.content.scrollDown()

    def update(self):
        self.content.update()

    def replaceContent(self, sender):
        self.content = sender.tag().replaceView(self.content)
        self.content.container.updateAll()

    def replaceSelf(self, sender):
        global modelTitle
        if sender.tag != None:
            self.content = sender.tag().replaceView(self)
            modelTitle = self.content.title
            self.content.container.updateAll()
            # self.content.updateAll()

    def hoverMouse(self, x, y):
        self.content.hoverMouse(x, y)


class MenuPage(ModelPage):
    def __init__(self):
        content = HStack([
            None,
            VStack([
                HStack([
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
                        self.createMenuButton(text="SVM", color=Color.blue, tag=self.createSVM),
                        self.createMenuButton(text="Neural Networks", color=Color.gray),
                        None
                    ], hideAllContainers=True)
                ]),
                None,
                self.createMenuButton(text="Compare Models", color=Color.green, tag=self.createComp)
            ], ratios=[0.8, 0.05, 0.15]),
            None
        ], ratios=[0.15, 0.7, 0.15])
        super().__init__(content=content, title="Select Machine Learning Model", includeTaskList=False)

    def createMenuButton(self, text, color, tag=None):
        return createButton(text=text, color=color, tag=tag, run=self.replaceSelf, hideAllContainers=True)

    def createDecisionTree(self):
        return ModelPage(content=IntroDTPage(), title="Decision Tree",
                         pages=[
            ("Intro", IntroDTPage),
            ("Example", ExampleDTPage),
            ("Improve", ExceriseDTPage),
            ("Coding", CodingDTPage)
            # ("More Info", InfoDTPage)
        ])

    def createKNN(self):
        return ModelPage(content=ExampleKNNPage(), title="KNN",
                         pages=[
            ("Intro", IntroKNNPage),
            ("Example", ExampleKNNPage),
            ("Coding", CodingKNNPage)
            # ("More Info", InfoKNNPage)
        ])

    def createLinear(self):
        return ModelPage(content=ExampleLinearPage(), title="Linear Regression",
                         pages=[
            ("Intro", IntroLinearPage),
            ("Linear", ExampleLinearPage),
            ("Quadratic", QuadLinearPage),
            # ("Subset", SubsetLinearPage),
            ("Coding", CodingLinearPage)
            # ("More Info", InfoLinearPage)
        ])

    def createLogistic(self):
        return ModelPage(content=ExampleLogisticPage(), title="Logistic Regression",
                         pages=[
            ("Intro", IntroLogisticPage),
            ("Example", ExampleLogisticPage),
            ("Coding", CodingLogisticPage),
            # ("More Info", InfoLogisticPage)
        ])

    def createSVM(self):
        return ModelPage(content=ExampleSVMPage(), title="Support Vector Machine",
                         pages=[
            ("Intro", IntroSVMPage),
            ("Example", ExampleSVMPage)
            # ("Coding", CodingLogisticPage),
            # ("More Info", InfoLogisticPage)
        ])

    def createComp(self):
        return ModelPage(content=CompPage(), title="Model Comparsions",
                         pages=[
            ("Home", CompPage),
            # ("Example", ExampleLogisticPage),
            # ("Coding", CodingLogisticPage),
            # ("More Info", InfoLogisticPage)
        ])

# =====================================================================
# BASE PAGE CLASSES
# =====================================================================


class CodingPage(SingleModel, ZStack):
    def __init__(self, codes, codingAddString, codingFilePath, codingExamplePath, **kwargs):
        self.codes = codes
        self.codingAddString = codingAddString
        self.codingFilePath = codingFilePath
        self.codingExamplePath = codingExamplePath

        ZStack.__init__(self, [
            VStack([
                self.createCodingHeader(),
                self.createCodingTable(),
                self.createCodingOptions()
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

    def incMethod(self, sender):
        pass

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
        self.leftStack = VStack(codeViews, name="left", containerArgs=[{"showEmpty": True}])
        self.rightStack = VStack([None] * len(self.codes), name="right", containerArgs=[{"showEmpty": True, "tag": code.order} for code in self.codes])

        self.codingTable = HStack([
            self.leftStack,
            self.rightStack
        ], ratios=[0.3, 0.7])
        # print("RUNNING CODE ROW:", len(self.codingTable.getView(1).containers))
        # for c in self.codingTable.getView(1).containers:
        # print("CODING ROW:", type(c), "Tag:", c.tag)
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
        return container.getParentStack().name == "left" or container.tag == view.tag.order

    def draggedView(self, view):

        stack = view.getParentStack()
        label = view.keyDown("label")
        isLeftStack = stack.name == "left"

        label.setFont(text=view.tag.label if isLeftStack else view.tag.line, fontSize=22)
        view.lock(lockedWidth=label.getWidth() + 60)

        success = True
        for stackView in self.rightStack.getViews():
            if stackView == None:
                success = False
                break
        self.codingRunRect.color = Color.green if success else Color.gray
        self.codingRunButton.isDisabled = not success

    def createFileNameView(self):
        return Label("File: " + self.table.fileName, fontSize=18)

    def createOpenSpreadsheetView(self):
        return Button([
            Rect(Color.green, cornerRadius=10),
            Label("Open Excel")
        ], run=hp.openFile, tag=self.table.filePath + ".csv", lockedWidth=200)

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
        self.codingScoreLabel.setFont(text=self.model.getScoreString())
        self.codingScoreLabel.container.updateAll()


# =====================================================================
# CUSTOM MODEL PAGES
# =====================================================================

# Decision Tree
class IntroDTPage(IntroView):
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
        super().__init__(label=Label(description))


class ExampleDTPage(SingleModel, ZStack):
    def __init__(self, **kwargs):
        SingleModel.__init__(self, **kwargs)
        self.setTable(Table(filePath="examples/decisionTree/dt_movie"))
        self.setModel(DecisionTree(table=self.table, testing=self.testingTable))

        ZStack.__init__(self, [
            HStack([
                VStack([
                    HStack([
                        TableView(model=self.model),
                        TreeRoom(model=self.model)

                    ], ratios=[0.6, 0.4]),
                    HeaderButtons(model=self.model)
                ], ratios=[0.9, 0.1])
            ]),
            TextboxView(textboxScript=[
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
            ], textboxAudioPath="dt_final/dt")
        ])


class ExceriseDTPage(SingleModel, ZStack):
    def __init__(self):
        SingleModel.__init__(self)
        self.setTable(Table(filePath="examples/decisionTree/dt_zoo"), partition=0.3)
        self.setModel(RandomForest(table=self.table, testingTable=self.testingTable))

        ZStack.__init__(self, [
            HStack([
                VStack([
                    HStack([
                        TreeList(model=self.model),
                        TableView(model=self.model),
                        TreeRoom(model=self.model),

                    ], ratios=[0.15, 0.55, 0.3]),
                    HeaderButtons(model=self.model)

                ], ratios=[0.9, 0.1])
            ])
        ])


class CodingDTPage(CodingPage):
    def __init__(self, **kwargs):
        self.setTable(Table(filePath="examples/decisionTree/dt_medical"))
        self.setModel(DecisionTree(table=self.table, testingTable=self.testingTable))

        # Codes
        codes = [
            Code("model = DecisionTreeClassifier()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="Add Tree", codingFilePath="assets/treeExample.py",
                         codingExamplePath="examples/decisionTree", **kwargs)


class InfoDTPage(InfoView):
    def __init__(self, **kwargs):
        files = [
            ("Generatation", "assets/decisiontree/GenerateDecisionTree.pdf"),
            ("Improvement", "assets/decisiontree/ImproveDecisionTree.pdf")
        ]
        super().__init__(files=files, **kwargs)


# KNN
class IntroKNNPage(IntroView):
    def __init__(self):
        description = ["Welcome to the KNN Introduction Page"]
        super().__init__(label=Label(description))


class ExampleKNNPage(MultiModel, ZStack):

    def __init__(self):
        MultiModel.__init__(self)

        self.setTable(Table(filePath="examples/knn/knn_iris", features=2), partition=0.3)
        self.addModel(KNN(table=self.table, testingTable=self.testingTable, drawTable=True))

        ZStack.__init__(self, [
            KNNGraphView(models=self.models, hasAxis=True, enableUserPts=True),
            TextboxView(textboxScript=[
                ("Welcome to the KNN Simulator!", 0, 0)
            ])
        ])


class CodingKNNPage(CodingPage):
    def __init__(self, **kwargs):
        self.setTable(Table(filePath="examples/knn/knn_iris"), partition=0.3)
        self.setModel(KNN(table=self.table, testingTable=self.testingTable, drawTable=True))
        # Codes
        codes = [
            Code("model = KNeighborClassifier()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="K: 3", codingFilePath="assets/treeExample.py", codingExamplePath="examples/knn", **kwargs)


class InfoKNNPage(InfoView):
    def __init__(self, **kwargs):
        files = [
            ("Bayes and KNN", "assets/knn/TeachingMaterialsKNN.pdf"),
            ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)

# Linear


class IntroLinearPage(IntroView):
    def __init__(self):
        description = ["Welcome to the Linear Regression Introduction Page"]
        super().__init__(label=Label(description))


class ExampleLinearPage(MultiModel, ZStack):
    def __init__(self):
        MultiModel.__init__(self)
        self.setTable(Table(filePath="examples/linear/linear_iris", features=1), partition=0.3)
        self.addModel(Linear(table=self.table, testingTable=self.testingTable, n=1, drawTable=True, isUserSet=True))
        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, n=1, drawTable=False, color=Color.blue))
        ZStack.__init__(self, [
            VStack([
                LinearGraphView(models=self.models, compModels=self.compModels, hasAxis=True, hoverEnabled=True)
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the Linear Regression Simulator!", 0, 0)
            ])
        ], hoverEnabled=True)
        # self.updateHeaderSelectionButtons()
        # print(view)


class QuadLinearPage(MultiModel, ZStack):
    def __init__(self):
        MultiModel.__init__(self)
        self.setTable(Table(filePath="examples/linear/linear_iris", features=1), partition=0.3)
        self.addModel(Linear(table=self.table, testingTable=self.testingTable, n=2, drawTable=True, isUserSet=True))
        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, n=2, drawTable=False, color=Color.blue))
        ZStack.__init__(self, [
            VStack([
                LinearGraphView(models=self.models, compModels=self.compModels, hasAxis=True, hoverEnabled=True)
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the Linear Regression Simulator!", 0, 0)
            ])
        ], hoverEnabled=True)
        # self.updateHeaderSelectionButtons()
        # print(view)


# class SubsetLinearPage(LinearPage):
#     def __init__(self):
#         items = [
#             VStack([
#                 LinearGraphView(model=self.model),
#                 self.createHeaderButtons()
#             ], ratios=[0.9, 0.1]),
#             self.createAddCompButton(),
#             self.createNextTextbox()  # must be last item
#         ]

#         super().__init__(textboxScript=[
#             ("Welcome to the Linear Regression Simulator!", 0, 0)
#         ], table=Table(filePath="examples/linear/linear_iris"), drawComp=True, item=items)
#         # self.updateHeaderSelectionButtons()


class CodingLinearPage(CodingPage):

    def __init__(self, **kwargs):
        self.setTable(Table(filePath="examples/linear/linear_iris", features=1), partition=0.3)
        self.setModel(Linear(table=self.table, testingTable=self.testingTable))
        # Codes
        codes = [
            Code("model = LinearRegression()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="--", codingFilePath="assets/treeExample.py", codingExamplePath="examples/linear", **kwargs)


class InfoLinearPage(InfoView):
    def __init__(self, **kwargs):
        files = [
            ("Linear Regression", "assets/linear/LinearRegression.pdf"),
            ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)


# Logistic
class IntroLogisticPage(IntroView):
    def __init__(self):
        description = ["Welcome to the Logisitic Regression Introduction Page"]
        super().__init__(label=Label(description))


class ExampleLogisticPage(MultiModel, ZStack):
    def __init__(self):
        MultiModel.__init__(self)
        self.setTable(Table(filePath="examples/logistic/logistic_sigmoid", constrainX=(0, 1 - 1e-5), constrainY=(0, 1 - 1e-5), features=1), partition=0.3)
        self.addModel(Logistic(table=self.table, testingTable=self.testingTable, drawTable=True, isUserSet=True))
        ZStack.__init__(self, [
            VStack([
                GraphView(models=self.models, hasAxis=True, hoverEnabled=True)
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the Logistic Regression Simulator!", 0, 0)
            ])
        ], hoverEnabled=True)


class CodingLogisticPage(CodingPage):
    def __init__(self, **kwargs):
        self.setTable(Table(filePath="examples/logistic/logistic_sigmoid", constrainX=(0, 1 - 1e-5), constrainY=(0, 1 - 1e-5), features=1), partition=0.3)
        self.setModel(Logistic(table=self.table, testingTable=self.testingTable, drawTable=False, isUserSet=False))
        # Codes
        codes = [
            Code("model = LogisticRegression()", "Load Model", 1),
            Code("data = pandas.read_csv('example.csv')", "Load Data", 1),
            Code("train, test = train_test_split(data, test_size=0.3)", "Split Data", 2),
            Code("model.fit(train,train['y'])", "Train Data", 3),
            Code("answer = model.predict(test)", "Run Test", 4),
            Code("return 100 * metrics.accuracy_score(test['y'], answer)", "Get Results", 5)
        ]
        super().__init__(codes=codes, codingAddString="--", codingFilePath="assets/treeExample.py", codingExamplePath="examples/linear", **kwargs)


class InfoLogisticPage(InfoView):
    def __init__(self, **kwargs):
        files = [
            # ("Linear Regression", "assets/linear/LinearRegression.pdf"),
            # ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)

# SVM


class IntroSVMPage(IntroView):
    def __init__(self):
        description = ["Welcome to the Support Vector Machine Introduction Page"]
        super().__init__(label=Label(description))


class ExampleSVMPage(MultiModel, ZStack):
    def __init__(self):
        MultiModel.__init__(self)
        self.setTable(Table(filePath="examples/svm/svm_iris", features=2), partition=0.3)  # , constrainX=(0, 1)
        self.addCompModel(SVM(table=self.table, testingTable=self.testingTable, drawTable=True))
        ZStack.__init__(self, [
            VStack([
                SVMGraphView(models=self.models, compModels=self.compModels, enableUserPts=False, hasAxis=True, hoverEnabled=False)
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the SVM Simulator!", 0, 0)
            ])
        ])


class CompPage(MultiModel, ZStack):
    def __init__(self):

        MultiModel.__init__(self)

        def x2(x):
            return x * x

        # svm_data = Data([{
        #     "type": "double",
        #     "x1": {
        #         "dist": "normal",
        #         "mean": 5,
        #         "std": 2
        #     },
        #     "y1": {
        #         "dist": "normal",
        #         "mean": 0,
        #         "std": 8
        #     },
        #     "x2": {
        #         "dist": "normal",
        #         "mean": 3,
        #         "std": 1
        #     },
        #     "y2": {
        #         "dist": "normal",
        #         "mean": 50,
        #         "std": 10
        #     },
        #     "func1": x2
        # }], labelValues=[-1, 1])
        data = Data([{
            "type": "single",
            "x": {
                "dist": "normal",
                "mean": 5,
                "std": 2
            },
            "y": {
                "dist": "normal",
                "mean": 0,
                "std": 8
            },
            "func": x2
        }])

        self.setTable(data.getTable(), partition=0.3)
        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, name="Linear", n=1, drawTable=True, alpha=1e-5))
        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, name="Quadratic", color=Color.blue, n=2, drawTable=False, alpha=1e-9))
        # self.addCompModel(Logistic(table=self.table, testingTable=self.testingTable, name="Logistic", color=Color.green, drawTable=False))
        ZStack.__init__(self, [
            VStack([
                GraphView(models=self.models, compModels=self.compModels, hasAxis=True)
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            TextboxView(textboxScript=[
                ("Welcome to the Model Comparator!", 0, 0)
            ])
        ])

        # self.models = [(KNN, {
        #         "k": 1,
        #         "table": data.training
        #     }), (Logistic, {
        #         "table": data.training
        #     })]
        # self.models = [model(**args) for model, args in self.modelClass]

        # modelCount = 3
        # runCount = 10
        # error = [[] for _ in range(modelCount)]

        # startTime = time()
        # for i in range(runCount):

        #     # Scenario 1
        #     data = Data(Dist.T, xFeatures=[
        #         Feature(mean=0, std=1),
        #         Feature(mean=1, std=1)
        #     ], yFeatures=[
        #         Feature(mean=0, std=1),
        #         Feature(mean=1, std=1)
        #     ], trainCount=50, testCount=100, p=0.25)

        # print(data.training.data)
        # print(data.testing.data)
        # data = Data(Dist.Normal, xFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], yFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], trainCount=20, testCount=20, p=0.0)

        # Scenario 2
        # data = Data(Dist.T, xFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], yFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], trainCount=50, testCount=50, p=0.0)

        # Scenario 3
        # data = Data(Dist.Normal, xFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], yFeatures=[
        #     Feature(mean=10, std=1),
        #     Feature(mean=0, std=0.5)
        # ], trainCount=50, testCount=50, p=0.5)

        #     self.models = [
        #         KNN(bestK=True, table=data.training, testingTable=data.testing),
        #         KNN(k=1, table=data.training, testingTable=data.testing),
        #         Logistic(table=data.training, testingTable=self.testingTable)
        #     ]

        #     # print("PREDICTION:", self.models[0].predictPoint(10, 10))

        #     for i in range(len(self.models)):
        #         error[i].append(self.models[i].error())

        #     print("Ran:", (i + 1), "/", runCount, end="\r")
        # print("Get Error Time:", round(time() - startTime, 2))
        # for i in range(len(self.models)):
        #     print("\nModel", i + 1)
        #     print("\tMean:", stat.mean(error[i]))
        #     print("\tSt Dev:", stat.stdev(error[i]))
        #     print("\tMin:", min(error[i]))
        #     print("\tMax:", max(error[i]))

        # import matplotlib.pyplot as plt

        # x = np.array([i for i in range(len(self.models))])
        # y = np.array([stat.mean(e) for e in error])
        # std = np.array([stat.stdev(e) for e in error])
        # # colors = [(model.color[0] / 255, model.color[1] / 255, model.color[2] / 255) for model in self.models]
        # # print(colors)
        # # ['red', 'green', 'blue', 'cyan', 'magenta']
        # plt.errorbar(x, y, std, linestyle='None', marker='.')
        # plt.show()

        # description = ["Welcome to the Comparsion Model Page"]
        # super().__init__(errorLabel=Label(description))

# =====================================================================
# Support Classes
# =====================================================================


class Code:

    def __init__(self, line, label, order):
        self.line = line
        self.label = label
        self.order = order


if __name__ == '__main__':
    pass
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    # page = CompPage()
