from graphics import *
import helper as hp
from table import *
from models import *
from random import shuffle
from math import sin, cos, pi
from elements import *
from comp import *
import statistics as stat
from time import time
from view import *


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
                self.createMenuButton(text="Compare Models", color=Color.green, tag=self.createComp),
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
            ("Linear", ExampleLinearPage),
            ("Quadratic", QuadLinearPage),
            # ("Subset", SubsetLinearPage),
            ("Coding", CodingLinearPage),
            ("More Info", InfoLinearPage)
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
        return ModelPage(content=ExampleLogisticPage(), title="Support Vector Machine",
                         pages=[
            # ("Intro", IntroLogisticPage),
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


class BasePage:
    def __init__(self, table):
        self.table, self.testingTable = table.partition(testing=0.3)

    def getView(self):
        pass


class SingleModelPage(BasePage):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model


class MultipleModelPage(BasePage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = []


class CodingPage(SingleModelPage):
    def __init__(self, codes, codingAddString, codingFilePath, codingExamplePath, **kwargs):
        super().__init__(**kwargs)
        self.codes = codes
        self.codingAddString = codingAddString
        self.codingFilePath = codingFilePath
        self.codingExamplePath = codingExamplePath

        # super().__init__(, table=table, items=items, hoverEnabled=False, **kwargs)

    def getView(self):
        return CodingView(model=self.model, codes=self.codes, codingAddString=self.codingAddString, codingFilePath=self.codingFilePath, codingExamplePath=self.codingExamplePath)

# =====================================================================
# MODEL PAGE CLASSES
# =====================================================================


class DTPage:
    def __init__(self, **kwargs):
        super().__init__(partition=0.3, **kwargs)
        self.model = DecisionTree(table=self.table, testing=self.testingTable)
        self.modelView = None
        # self.models.append(self.model)
        # if self.table.filePath:
        #     self.bagIndex = 0

    def createTreeRoomView(self):
        treeViewContainer = self.modelView.container if self.modelView != None else None
        self.modelView = Points(pts=self.model.getPts())
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
        if self.model.hasChildren():
            self.goToIndexTree(index=sender.tag)
            self.table = self.model.getTable()
            self.updateTableView()
            self.updateHeaderSelectionButtons()
            self.modelBranch.getContainer().updateAll()
            self.tableView.container.updateAll()
            # self.updateContainers()

    def goBack(self, sender):
        if not self.model.isRoot():
            self.goBackTree()
            self.table = self.model.getTable()
            self.updateTableView()
            self.updateHeaderSelectionButtons()
            self.modelBranch.getContainer().updateAll()
            self.tableView.container.updateAll()

    def saveTree(self, sender):
        accuracy = round(self.model.test(self.localTestTable) * 100)
        self.model.modelTest(self.finalTestTable)

        bagItem = ZStack([
            Rect(color=Color.steelBlue),
            Label(text="T:{} [{}%]".format(len(self.models), accuracy), fontSize=20)
        ], dy=1)
        bagItem.setContainer(self.bag[self.bagIndex])
        self.bagIndex += 1

        self.totalScoreLabel.setFont(text="Total: [{}%]".format(round(100 * self.getTotalScore())))
        self.table, self.localTestTable = self.trainTable.partition()

        self.model = DecisionTree(table=self.table)
        self.models.append(self.model)
        self.createTreeRoomView()
        self.updateTableView()

        self.bag.updateAll()
        self.modelView.container.updateAll()
        self.tableView.container.updateAll()
        self.updateHeaderSelectionButtons()

    def splitTree(self, column):
        self.model.add(column=column)
        container = self.modelBranch.getContainer()
        prevStack = type(self.modelBranch.disjoint.keyUp("div"))
        stack = VStack if prevStack != VStack else HStack
        label = [Button(Label(text="{}:{}".format(self.model.getParentColumn(), self.model.getValue()), fontSize=20,
                              color=Color.white, dx=-0.95, dy=-1), run=self.goBack)] if self.model.getParent() != None else []
        treeChildren = self.model.getChildren()
        totalClassCount = len(treeChildren)
        for child in treeChildren:
            totalClassCount += child.table.classCount

        self.modelView = ZStack(items=label + [
            stack(items=[
                Button(Points(pts=self.model.getChild(i).table.getPts(), isConnected=False), run=self.goForward, tag=i) for i in range(len(treeChildren))
            ], ratios=[
                (child.table.classCount + 1) / totalClassCount for child in treeChildren
            ], border=20 if self.model.getParent() != None else 0, keywords="div")
        ], keywords=["z"])
        self.modelBranch.setView(view=self.modelView)

    def removeNodeTree(self):
        self.model.remove()
        self.modelView = self.createDotViews(self.model.curr)
        self.modelBranch.setView(view=self.modelView)

    def goBackTree(self):
        self.model.goBack()
        self.modelView = self.modelBranch.disjoint.keyUp("z")
        self.modelBranch.move(self.modelView)

    def goToIndexTree(self, index):
        self.model.go(index=index)
        container = self.modelBranch.view.keyDown("div")[index]
        stack = container.keyDown("z")
        self.modelView = stack if stack != None else container.keyDown("dotStack")
        self.modelBranch.move(self.modelView)

    def selectColumn(self, sender):
        super().selectColumn(sender)
        rect = sender.getCousin(0)
        if sender.isOn:
            self.splitTree(column=self.model.getColName(sender.tag))
            rect.strokeColor = Color.white
        else:
            self.model.remove()
            rect.strokeColor = Color.gray
        self.modelBranch.getContainer().updateAll()


class KNNPage:
    def __init__(self, table, testingTable=None, **kwargs):
        super().__init__(table=table, testingTable=testingTable, model=KNN(table=table, testingTable=testingTable, drawTable=True), **kwargs)

    def clickGraph(self, sender):
        super().clickGraph(sender)
        if self.userPts != None:
            x = [
                hp.map(sender.lastClickX, sender.x, sender.x + sender.getWidth(), self.models[0].minX1, self.models[0].maxX1),
                hp.map(sender.lastClickY, sender.y, sender.y + sender.getHeight(), self.models[0].minX2, self.models[0].maxX2),
            ]
            pred = self.models[0].predict(x)
            self.userPts.setColor(-1, self.table.classColors[pred])

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


class LinearPage:
    def __init__(self, **kwargs):
        super().__init__(partition=0.3, hasAxis=True, features=1, **kwargs)
        self.addModel(Linear(table=self.table, testingTable=self.testingTable, n=2, drawTable=True, isUserSet=True))


class LogisticPage:
    def __init__(self, **kwargs):
        super().__init__(partition=0.3, hasAxis=True, features=1, constrainX=(0, 1 - 1e-5), constrainY=(0, 1 - 1e-5), **kwargs)
        self.addModel(Logistic(table=self.table, testingTable=self.testingTable, drawTable=True, isUserSet=True))


class SVMPage:
    def __init__(self, **kwargs):
        super().__init__(partition=0.3, hasAxis=True, features=1, **kwargs)
        self.models.append(SVM(table=self.table))


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


class ExampleDTPage(ZStack):
    def __init__(self, **kwargs):
        self.table = Table(filePath="examples/decisionTree/movie")
        self.model = DecisionTree(table=self.table, testing=self.testingTable)
        items = [
            HStack([
                VStack([
                    HStack([
                        TableView(),
                        self.createTreeRoomView()

                    ], ratios=[0.6, 0.4]),
                    self.createHeaderButtons()
                ], ratios=[0.9, 0.1])
            ]),
            self.createNextTextbox()
        ]
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
        ], textboxAudioPath="dt_final/dt", items=items)


class ExceriseDTPage(DTPage):
    def __init__(self):
        items = [
            HStack([
                VStack([
                    HStack([
                        self.createTreeListView(),
                        self.tableView,
                        self.createTreeRoomView()

                    ], ratios=[0.15, 0.55, 0.3]),
                    self.createHeaderButtons()

                ], ratios=[0.9, 0.1])
            ])
        ]

        super().__init__(table=Table(filePath="examples/decisionTree/zoo"), items=items)
        self.models = []  # different from GraphView


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
        table = Table(filePath="examples/decisionTree/medical")
        super().__init__(table=table, model=DecisionTree(table=table), codes=codes, codingAddString="Add Tree", codingFilePath="assets/treeExample.py",
                         codingExamplePath="examples/decisionTree", **kwargs)

    def incMethod(self, sender):
        if len(self.models) >= 100:
            self.models = []
            self.codingAddLabel.setFont("Add Trees")

        else:
            for _ in range(10):
                train, test = self.trainTable.partition()
                self.model = DecisionTree(table=train)
                self.addModel(self.model)
            self.codingAddLabel.setFont("Trees: {}".format(len(self.models)))


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


class ExampleKNNPage(KNNPage):
    def __init__(self):
        items = [
            self.createGraphView(),
            self.createNextTextbox()
        ]
        super().__init__(textboxScript=[
            ("Welcome to the KNN Simulator!", 0, 0)
        ], table=Table(filePath="examples/knn/iris"), hasAxis=True, enableUserPts=True, items=items)  # examples/linear/iris

        # print(view)

    def hoverMouse(self, mouseX, mouseY):
        if self.hoverEnabled and self.isWithin(mouseX, mouseY):
            pass
            # while self.modelView.peekView().name == "highlight":
            #     self.modelView.popView()
            # x, y = hp.map(mouseX - self.modelView.x, 0.0, self.modelView.getWidth(), -1.0, 1.0), hp.map(mouseY - self.modelView.y, 0.0, self.modelView.getHeight(), -1.0, 1.0)
            # for _, index, dx, dy in self.model.getNeighbor(x, y):
            #     self.modelView.addView(
            #         Ellipse(color=self.model.classColors[self.table.loc[index][self.table.targetName]], dx=dx, dy=dy, lockedWidth=20, lockedHeight=20, name="highlight")
            #     )
            # self.updateAll()


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
        super().__init__(codes=codes, codingAddString="K: 3", table=Table(filePath="examples/knn/iris"), codingFilePath="assets/treeExample.py", examplePath="examples/knn", **kwargs)


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


class ExampleLinearPage(LinearPage):
    def __init__(self):
        items = [
            VStack([
                self.createGraphView()
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]
        super().__init__(textboxScript=[
            ("Welcome to the Linear Regression Simulator!", 0, 0)
        ], table=Table(filePath="examples/linear/iris"), items=items)

        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, color=Color.blue, n=2))

        # self.updateHeaderSelectionButtons()
        # print(view)


class QuadLinearPage(LinearPage):
    def __init__(self):
        items = [
            VStack([
                self.createGraphView(),
                self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]

        super().__init__(textboxScript=[
            ("Welcome to the Linear Regression Simulator!", 0, 0)
        ], table=Table(filePath="examples/linear/test"), items=items)
        self.model.n = 2
        self.addCompModel(Linear(table=self.table, testingTable=self.testingTable, color=Color.blue, n=2))
        # self.updateHeaderSelectionButtons()
        # print(view)


class SubsetLinearPage(LinearPage):
    def __init__(self):
        items = [
            VStack([
                self.createGraphView(),
                self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]

        super().__init__(textboxScript=[
            ("Welcome to the Linear Regression Simulator!", 0, 0)
        ], table=Table(filePath="examples/linear/iris"), drawComp=True, item=items)
        # self.updateHeaderSelectionButtons()


class CodingLinearPage(LinearPage, CodingPage):
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
        super().__init__(codes=codes, codingAddString="--", table=Table(filePath="examples/linear/iris", features=1),
                         codingFilePath="assets/treeExample.py", examplePath="examples/linear", **kwargs)


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


class ExampleLogisticPage(LogisticPage):
    def __init__(self):
        items = [
            VStack([
                self.createGraphView()
                # ,
                # self.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            # self.createAddCompButton(),
            self.createNextTextbox()  # must be last item
        ]
        super().__init__(textboxScript=[
            ("Welcome to the Logistic Regression Simulator!", 0, 0)
        ], table=Table(filePath="examples/logistic/sigmoid"), items=items)
        # self.updateHeaderSelectionButtons()
        # print(view)


class CodingLogisticPage(LogisticPage, CodingPage):
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
        super().__init__(codes=codes, codingAddString="--", table=Table(filePath="examples/linear/iris", features=1),
                         codingFilePath="assets/treeExample.py", examplePath="examples/linear", **kwargs)


class InfoLogisticPage(InfoView):
    def __init__(self, **kwargs):
        files = [
            # ("Linear Regression", "assets/linear/LinearRegression.pdf"),
            # ("Cross Validation", "assets/general/CrossValidation.pdf")
        ]
        super().__init__(files=files, **kwargs)

# SVM


class ExampleSVMPage(SVMPage):
    def __init__(self):
        items = [
            self.createGraphView(),
            self.createNextTextbox()
        ]
        super().__init__(textboxScript=[
            ("Welcome to the SVM Simulator!", 0, 0)
        ], table=Table(filePath="examples/svm/iris"), items=items)  # examples/linear/iris


class CompPage(IntroView):
    def __init__(self):

        # self.models = [(KNN, {
        #         "k": 1,
        #         "table": data.training
        #     }), (Logistic, {
        #         "table": data.training
        #     })]
        # self.models = [model(**args) for model, args in self.modelClass]

        modelCount = 3
        runCount = 10
        error = [[] for _ in range(modelCount)]

        startTime = time()
        for i in range(runCount):

            # Scenario 1
            data = Data(Dist.T, xFeatures=[
                Feature(mean=0, std=1),
                Feature(mean=1, std=1)
            ], yFeatures=[
                Feature(mean=0, std=1),
                Feature(mean=1, std=1)
            ], trainCount=50, testCount=100, p=0.25)

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

            self.models = [
                KNN(bestK=True, table=data.training, testingTable=data.testing),
                KNN(k=1, table=data.training, testingTable=data.testing),
                Logistic(table=data.training, testingTable=self.testingTable)
            ]

            # print("PREDICTION:", self.models[0].predictPoint(10, 10))

            for i in range(len(self.models)):
                error[i].append(self.models[i].error())

            print("Ran:", (i + 1), "/", runCount, end="\r")
        print("Get Error Time:", round(time() - startTime, 2))
        for i in range(len(self.models)):
            print("\nModel", i + 1)
            print("\tMean:", stat.mean(error[i]))
            print("\tSt Dev:", stat.stdev(error[i]))
            print("\tMin:", min(error[i]))
            print("\tMax:", max(error[i]))

        import matplotlib.pyplot as plt

        x = np.array([i for i in range(len(self.models))])
        y = np.array([stat.mean(e) for e in error])
        std = np.array([stat.stdev(e) for e in error])
        # colors = [(model.color[0] / 255, model.color[1] / 255, model.color[2] / 255) for model in self.models]
        # print(colors)
        # ['red', 'green', 'blue', 'cyan', 'magenta']
        plt.errorbar(x, y, std, linestyle='None', marker='.')
        plt.show()

        description = ["Welcome to the Comparsion Model Page"]
        super().__init__(errorLabel=Label(description))

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


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    page = CompPage()
