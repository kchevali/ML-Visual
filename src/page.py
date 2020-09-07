from graphics import *
import helper as hp
from table import Table
from decisionTree import DecisionTree
from controller import *


def createMouseDebug():
    return ZStack([
        Rect(color=Color.white, keywords="mRect", border=0),
        Label(text="", fontSize=15, color=Color.black, keywords="text")
    ], lockedWidth=80, lockedHeight=20, dx=-1, dy=1)


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
            Table(filePath="examples/shape", createView=createView)
        ]
        super().__init__(items)


class MainPage(ZStack):
    def __init__(self):
        items = [
            MenuPage()
        ]
        super().__init__(items)


class ModelPage(ZStack):
    def __init__(self, content, title, pages=[]):

        self.content = content
        self.title = title
        self.pages = pages

        items = [
            VStack([
                self.createTitle(),
                HStack([
                    self.createTaskList(),
                    self.content
                ], ratios=[0.15, 0.85])
            ], ratios=[0.08, 0.92])

        ]
        super().__init__(items)

    def createTitle(self):
        return ZStack([
            Rect(color=Color.steelBlue),
            Button(Label("<", dx=-1, offsetX=20), run=self.replaceSelf, tag=MenuPage) if type(self) != MenuPage else None,
            Label(self.title)
        ])

    def createTaskList(self):
        return VStack([
            Button([
                Rect(color=Color.orange, cornerRadius=10),
                Label(task)
            ], tag=page, run=self.replaceContent) for task, page in self.pages
        ], ratios=[1 / (len(self.pages) * 2) for _ in range(len(self.pages))])

    def canDragView(self, view, container):
        return self.content.canDragView(view=view, container=container)

    def scrollUp(self):
        self.content.scrollUp()

    def scrollDown(self):
        self.content.scrollDown()

    def replaceContent(self, sender):
        self.content = sender.tag().replaceView(self.content)
        self.content.container.updateAll()

    def replaceSelf(self, sender):
        self.content = sender.tag().replaceView(self)
        self.content.container.updateAll()


class MenuPage(ModelPage):
    def __init__(self):
        content = VStack([
            None,
            HStack([
                None,
                self.createButton(text="Decision Tree", color=Color.blue, tag=self.createDecisionTree),
                self.createButton(text="KNN", color=Color.red, tag=self.createKNN),
                None
            ]), None
        ])
        super().__init__(content=content, title="Select Model")

    def createButton(self, text, color, tag):
        return Button([
            Rect(color=color, cornerRadius=10),
            Label(text=text)
        ], tag=tag, name=text, run=self.replaceSelf)

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
        return ModelPage(content=IntroDTPage(), title="KNN",
                         pages=[
            ("Intro", IntroDTPage),
            ("Example", ExampleKNNPage),
            ("Coding", CodingDTPage),
            ("More Info", InfoDTPage)
        ])


class IntroDTPage(ZStack):
    def __init__(self):
        views = [

            Label("Description", fontSize=56, dx=-1, dy=-1, offsetX=10, offsetY=10),

            Label(
                """    A tree has many analogies in real life and turns out that it has influenced
a wide area of machine learning, covering both classification and regression.

    Tree-based methods involve stratifying or segmenting the predictor space
into a number of simple regions. Since the set of splitting rules used to
segment the predictor space can be summarized in a tree, these types of
approaches are known as decision tree methods. The structure of a decision
tree includes:
    1) internal nodes corresponding to attributes (features);
    2) leaf nodes corresponding to the classification outcome;
    3) edge denoting the assignment of the attribute.""")


        ]
        super().__init__(views)


class ExampleDTPage(ZStack):
    def __init__(self):
        self.ctr = Controller(filePath="examples/movie", partition=False, textboxScript=[
            ("Welcome to the Decision Tree Simulator!", 0, 0),
            (
                """To begin we will use a Decision Tree to
analyze movie data. The objective of the
model is to predict if movies would be
liked or disliked based on type, length,
and other characteristics.""", 0.8, 0
            ),
            (
                """Lets start by splitting the data shown on
the right into separate groups of the
same color""", -0.5, 0),
            ("Click on director to split the data into 3 groups", -0.8, 0),
            ("Next, click on director lass to subdivide the group", -0.8, 0),
            ("Finally, click on length to complete the tree", -0.8, 0),
            ("To show the full tree click on group name on the top", -0.8, 0),
            ("Congratulations on completing the tutorial", 0, 0)
        ], textboxStack=self)
        views = [
            HStack([
                VStack([
                    # ===============================================================================================
                    # TABLE & MODEL
                    # ===============================================================================================
                    HStack([
                        self.ctr.createTableView(),
                        self.ctr.createTreeRoomView()

                    ], ratios=[0.6, 0.4]),

                    # ===============================================================================================
                    # HYPERPARAMETERS
                    # ===============================================================================================
                    self.ctr.createHeaderButtons()

                ], ratios=[0.9, 0.1])
            ]),

            # ===============================================================================================
            # TEXT BOX OVERLAY
            # ===============================================================================================
            self.ctr.createNextTextbox()
        ]
        # print(view)
        super().__init__(views)

    def scrollUp(self):
        self.ctr.shiftTable(dy=1)

    def scrollDown(self):
        self.ctr.shiftTable(dy=-1)


class ExceriseDTPage(ZStack):
    def __init__(self):
        self.ctr = Controller(filePath="examples/zoo")
        views = [
            HStack([
                VStack([
                    # ===============================================================================================
                    # TABLE & MODEL
                    # ===============================================================================================
                    HStack([
                        self.ctr.createTreeListView(),
                        self.ctr.createTableView(),
                        self.ctr.createTreeRoomView()

                    ], ratios=[0.15, 0.55, 0.3]),

                    # ===============================================================================================
                    # HYPERPARAMETERS
                    # ===============================================================================================
                    self.ctr.createHeaderButtons()

                ], ratios=[0.9, 0.1])
            ])
        ]
        # print(view)
        super().__init__(views)

    def scrollUp(self):
        self.ctr.shiftTable(dy=1)

    def scrollDown(self):
        self.ctr.shiftTable(dy=-1)


class CodingDTPage(ZStack):
    def __init__(self):
        self.ctr = Controller(filePath="examples/medical")
        items = [
            HStack([
                VStack([
                    self.ctr.createCodeTitle(),
                    HStack([
                        self.ctr.createCodeView(),
                        VStack([None] * len(self.ctr.codes), keywords="ans")

                    ], ratios=[0.3, 0.7])
                ], ratios=[0.1, 0.9]),
                self.ctr.createFileExplorerView()
            ], ratios=[0.85, 0.15])

        ]
        super().__init__(items)

    def canDragView(self, view, container):
        srcStack = view.getParentStack()
        destStack = container.getParentStack()
        if srcStack.findKey("ans") or destStack.findKey("ans"):
            label = view.keyDown("label")
            if destStack.findKey("ans"):
                view.lock(lockedWidth=400)
                label.setFont(text=view.tag.line, fontSize=22)
            else:
                view.lock(lockedWidth=200)
                label.setFont(text=view.tag.label, fontSize=32)

            success = True

            destContainers = destStack.containers
            for i in range(1, len(destContainers)):
                codeStack = destContainers[i].view if destContainers[i] != container else view
                prevStack = destContainers[i - 1].view if destContainers[i - 1] != container else view
                if codeStack == None or prevStack == None or codeStack.tag.order < prevStack.tag.order:
                    success = False
                    break
            self.ctr.runButtonRect.color = Color.green if success else Color.gray
            self.ctr.runButton.isDisabled = not success

            return True


class InfoDTPage(ZStack):
    def __init__(self):
        self.ctr = Controller()
        items = [self.ctr.createInfoDTViews()]
        super().__init__(items)


class ExampleKNNPage(ZStack):
    def __init__(self):
        self.ctr = Controller(filePath="examples/car", partition=False, textboxScript=[
            ("Welcome to the KNN Simulator!", 0, 0)
        ], textboxStack=self)
        views = [
            VStack([
                self.ctr.createTableView(),
                self.ctr.createHeaderButtons()
            ], ratios=[0.9, 0.1]),
            self.ctr.createNextTextbox()
        ]
        # print(view)
        super().__init__(views)
