from gui import g
from time import time
import helper as hp
import pygame as pg
import pygame.gfxdraw as pgx
from math import inf
from random import uniform
from colorsys import hsv_to_rgb
from collections import deque

# ===========================================================
# FRAMES
# ===========================================================


class Frame:

    idCounter = 0

    def __init__(self, name="", tag=0, keywords="", dx=0.0, dy=0.0, offsetX=0.0, offsetY=0.0, isHidden=False, hideContainer=False, hideAllContainers=False, isDraggable=False, isDisabled=False, container=None):
        # INPUTS
        self.name = name
        self.tag = tag
        self.keywords = keywords
        self.dx = dx
        self.dy = dy
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.isHidden = isHidden
        self.isDisabled = isDisabled
        self.hideContainer = hideContainer
        self.hideAllContainers = hideAllContainers
        self.isDraggable = isDraggable

        self.container = container  # value set at the end

        # INITIAL VALUES
        self._width = 0.0
        self._height = 0.0
        self.minWidth = 0.0
        self.minHeight = 0.0
        self.scrollX = 0.0
        self.scrollY = 0.0
        self.border = 0.0  # unused by fixed frames
        self.isExpanded = False
        self.setup = []
        self.id = Frame.idCounter
        Frame.idCounter += 1

        # CONSTS CANNOT CHANGE
        self.isWidthLocked = True
        self.isHeightLocked = True
        self.canHold = False

        self._setXY(x=0.0, y=0.0)
        if self.container != None:
            self.setContainer(container=self.container)
        if type(self.keywords) != list:
            self.keywords = [self.keywords]

    def _setXY(self, x, y):
        self.x = x
        self.y = y
        self.pos = (self.x, self.y)

    def setAlignment(self, dx=None, dy=None):
        if dx != None:
            self.dx = dx
        if dy != None:
            self.dy = dy

    def addInstruction(self, method, args):
        self.setup.append((method, args))

    def delink(self, allowButtonUpdate=True):
        if self.container != None:
            self.container.view = None
            if self.container.isButton() and allowButtonUpdate:
                self.container.viewBackup = None
            self.container = None

    def setContainer(self, container, allowButtonUpdate=True):
        self.delink(allowButtonUpdate=allowButtonUpdate)
        if container != None:
            if container.view != None:
                container.view.delink(allowButtonUpdate=allowButtonUpdate)
            self.container = container
            self.container.view = self
            if self.container.isButton() and allowButtonUpdate:
                self.container.viewBackup = self

    def replaceView(self, oldView):
        self.setContainer(container=oldView.container)
        return self

    def getWidth(self):
        return max(self._width, self.minWidth)

    def getHeight(self):
        return max(self._height, self.minHeight)

    def getParentStack(self):
        if self.container:
            return self.container if self.container.isStack() else self.container.getParentStack()

    def getCousin(self, key):
        stack = self.getParentStack()
        if stack:
            return stack.getView(key)

    def updateFrame(self):
        if self.container != None:
            self._setXY(x=self.container.x + self.offsetX + (self.dx + 1.0) * (self.container.getWidth() - self.getWidth()) / 2.0,
                        y=self.container.y + self.offsetY + (self.dy + 1.0) * (self.container.getHeight() - self.getHeight()) / 2.0)
            # print("UF:", type(self), self.id, self.x, self.y)

    def updateAll(self):
        for view in self.getLeafOrder():
            view.updateMinimumSizes()
        self.updateDown()

    def doesFit(self):
        return self.container == None or self.container.getWidth() == 0 or self.container.getHeight() == 0 or (self.getWidth() <= self.container.getWidth() and self.getHeight() <= self.container.getHeight())

    def isWithin(self, x, y):
        return x >= self.x and x <= self.x + self.getWidth() and y >= self.y and y <= self.y + self.getHeight()

    def updateMinimumSizes(self):
        if self.canHold:
            self.minWidth = self._width if self.isWidthLocked else 0.0
            self.minHeight = self._height if self.isHeightLocked else 0.0

    def getLeafOrder(self):
        yield self

    def getRootOrder(self):
        q = deque()
        yield from self.findRootOrder(q)
        while q:
            yield from q.popleft().findRootOrder(q)

    def findRootOrder(self, q):
        yield self

    def updateDown(self):
        self.updateFrame()

    def findKey(self, keyword):
        return self if keyword in self.keywords else None

    def keyUp(self, keyword, excludeSelf=False):
        if not excludeSelf:
            obj = self.findKey(keyword)
            if obj != None:
                return obj
        if self.container != None:
            return self.container.keyUp(keyword)

    def keyDown(self, keyword, excludeSelf=False):
        for obj in self.getRootOrder():
            if obj.findKey(keyword) != None and (not excludeSelf or self != obj):
                return obj

    def isContainer(self):
        return isinstance(self, Container)

    def isStack(self):
        return isinstance(self, Stack)

    def isButton(self):
        return isinstance(self, Button)

    def getID(self):
        return (self.keywords[0] if len(self.keywords) == 1 else self.keywords) if self.keywords[0] else self.id

    def _setSize(self, width, height):
        """
        Records size of frame.
        """
        self._width = width
        self._height = height

    def isClicked(self, x, y):
        return self.isWithin(x, y) and not self.isDisabled

    def clicked(self, x, y):
        if self.isClicked(x, y):
            return self

    def setSize(self, width=None, height=None):
        "Empty Method"

    def getSize(self):
        return (self.getWidth(), self.getHeight())

    def display(self):
        "Empty Method"
        # if not self.isHidden:
        #     pass
        # pg.draw.rect(g, Color.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)

    def get(self, key):
        return self.__dict__[key]


class ResizableFrame(Frame):
    def __init__(self, lockedWidth=None, lockedHeight=None, border=0, **args):
        super().__init__(**args)
        self.isWidthLocked = False
        self.isHeightLocked = False
        self.lock(lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.border = border

    def setSize(self, width=None, height=None):
        if width != None and not self.isWidthLocked:
            self._width = width
        if height != None and not self.isHeightLocked:
            self._height = height

    def lock(self, lockedWidth=None, lockedHeight=None):
        """
        Lock the width or height of a resizable frame
            Num   -> Lock length at num
            False -> Remove lock
            None  -> No effect
        """
        if lockedWidth != None:
            self.isWidthLocked = lockedWidth != False
            if self.isWidthLocked:
                self._width = lockedWidth  # Cannot be 0
                self.minWidth = lockedWidth

        if lockedHeight != None:
            self.isHeightLocked = lockedHeight != False
            if self.isHeightLocked:
                self._height = lockedHeight  # Cannot be 0
                self.minHeight = lockedHeight

    def setBorder(self, border):
        self.border = border


class Color(ResizableFrame):

    white = (255, 255, 255)
    red = (255, 59, 48)
    pink = (255, 45, 85)
    orange = (255, 149, 0)
    green = (76, 217, 100)
    yellow = (255, 204, 0)
    pink = (255, 51, 153)
    blue = (0, 122, 255)
    steelBlue = (70, 130, 180)
    lightSteelBlue = (176, 196, 222)
    lightBlue = (135, 206, 250)
    cadetBlue = (95, 158, 160)
    tealBlue = (90, 200, 250)
    medBlue = (0, 0, 205)
    darkBlue = (72, 61, 139)
    midnightBlue = (25, 25, 112)
    purple = (88, 86, 214)
    gray = (150, 150, 150)
    darkGray = (40, 40, 40)
    black = (0, 0, 0)
    backgroundColor = ((50, 50, 50))

    def __init__(self, color, **args):
        super().__init__(**args)
        self.color = color

    def display(self):
        if not self.isHidden:
            super().display()
            pg.draw.rect(g, self.color, (self.x, self.y, self.getWidth(), self.getHeight()))

    def __str__(self, indent=""):
        return "Color:{}".format(self.getID())

    @staticmethod
    def calmColor(hue):
        r, g, b = hsv_to_rgb(hue, 0.54, 0.8)
        return (r * 255, g * 255, b * 255, 0)


class Shape(Color):

    def __init__(self, strokeColor=None, strokeWidth=None, **args):
        super().__init__(**args)
        self.strokeColor = strokeColor
        self.strokeWidth = None if self.strokeColor == None else strokeWidth


class Rect(Shape):

    def __init__(self, color, cornerRadius=0, border=10, **args):
        super().__init__(color=color, border=border, **args)
        self.cornerRadius = cornerRadius

    def display(self):
        if not self.isHidden:
            # print("RECT:", self.pos, self.getSize())
            # print("RECTs")
            # if self.color :
            #     hp.draw_bordered_rounded_rect(surface=g, color=self.color, rect=(self.x, self.y, self.getWidth(), self.getHeight()), width=0, border_radius=self.cornerRadius)
            # if self.storkeColor  and self.strokeWidth > 0:
            # print("RECT:", (self.x, self.y, self.getWidth(), self.getHeight()))
            hp.draw_bordered_rounded_rect(g, rect=(self.x, self.y, self.getWidth(), self.getHeight()), color=self.color,
                                          border_color=self.strokeColor, corner_radius=self.cornerRadius, border_thickness=self.strokeWidth)

    def __str__(self, indent=""):
        return "Rect:{}".format(self.getID())


class Ellipse(Shape):

    def display(self):
        if not self.isHidden:
            # x, y, w, h = tuple(map(int, [self.x, self.y, self.getWidth() / 2, self.getHeight() / 2]))
            # x += w
            # y += h
            # print("Color:", self.color, self.color is not None)
            if self.color is not None:
                pg.draw.ellipse(g, self.color, (self.x, self.y, self.getWidth(), self.getHeight()))
            # if self.strokeColor and self.strokeWidth > 0:
            #     pgx.aaellipse(g, x, y, w, h, self.strokeColor)

    def __str__(self, indent=""):
        return "Ellipse:{}".format(self.getID())


class Label(Frame):
    # border has no effect on
    fontCache = {}

    def __init__(self, text, fontName="Comic Sans MS", fontSize=32, color=Color.white, autoFontSize=False, **args):
        super().__init__(**args)
        self.autoFontSize = autoFontSize
        self.setFont(text=text, fontName=fontName, fontSize=fontSize, color=color)

    def setFont(self, text=None, fontName=None, fontSize=None, color=None):
        if text != None:
            self.text = text
        if fontName != None:
            self.fontName = fontName
        if fontSize != None:  # and not self.autoFontSize:
            self.fontSize = fontSize
        if color != None:
            self.color = color

        fontKey = self.fontName + str(self.fontSize)
        if fontKey in Label.fontCache:
            self.font = Label.fontCache[fontKey]
        else:
            self.font = pg.font.SysFont(self.fontName, self.fontSize)
            Label.fontCache[fontKey] = self.font

        self.render(self.text)
        self.threeDots = False

    def render(self, text):
        lines = text.splitlines()
        self.surfaces = []
        width = 0
        for i, line in enumerate(lines):
            self.surfaces.append(self.font.render(line, True, self.color))
            lineWidth, lineHeight = self.font.size(line)
            width = max(width, lineWidth)
        self._setSize(width, len(lines) * self.fontSize - 5)

    def updateFrame(self):
        super().updateFrame()
        if not self.threeDots and not self.doesFit():
            self.threeDots = True
            self.render("...")

    def display(self):
        if not self.isHidden:
            super().display()
            for i, surface in enumerate(self.surfaces):
                g.blit(surface, (self.x, self.y + self.fontSize * i))

    def __str__(self, indent=""):
        return "Label:'{}'".format(self.text)

# ===========================================================
# CONTAINERS
# ===========================================================


class Holder(ResizableFrame):

    def __init__(self, view=None, **args):
        super().__init__(**args)
        self.view = view
        self.canHold = True
        if view != None:
            view.setContainer(container=self)

    def updateFrame(self):
        super().updateFrame()
        if self.view != None:
            self.view.setSize(width=self.getWidth() - 2 * self.view.border, height=self.getHeight() - 2 * self.view.border)

    def display(self):
        if not self.isHidden and self.view != None:
            super().display()
            self.view.display()

    def clicked(self, x, y):
        if self.isClicked(x, y):
            if self.isDraggable:
                return self
            clickedObj = self.view.clicked(x, y) if self.view != None else None
            return clickedObj if clickedObj != None else self

    def updateDown(self):
        super().updateDown()
        if self.view != None:
            self.view.updateDown()

    def getLeafOrder(self):
        if self.view != None:
            yield from self.view.getLeafOrder()
        yield self

    def findRootOrder(self, q):
        yield self
        if self.view != None:
            q.append(self.view)

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        if self.view != None:
            self.minWidth = max(self.minWidth, self.view.minWidth)
            self.minHeight = max(self.minHeight, self.view.minHeight)

    def __str__(self, indent=""):
        return "<{}>".format(self.view.__str__(indent=indent) if self.view != None else None)


class Container(Holder):

    def __init__(self, ratioX=1.0, ratioY=1.0, **args):
        super().__init__(**args)
        self.ratioX = ratioX
        self.ratioY = ratioY

    def updateRatios(self, ratioX=None, ratioY=None):
        if ratioX != None:
            self.ratioX = ratioX
        if ratioY != None:
            self.ratioY = ratioY

    def setContainer(self, container):
        self.container = container

    def delink(self):
        raise Exception("Cannot delink a container")

    def display(self):
        if not self.isHidden:
            if self.view == None or (not self.view.hideContainer and (self.container == None or not self.container.hideAllContainers)):
                pg.draw.rect(g, Color.darkGray, (self.x, self.y, self.getWidth(), self.getHeight()), 2)
            super().display()

# ===========================================================
# STACKS
# ===========================================================


class Stack(ResizableFrame):

    # init args have default values for ZStack()
    def __init__(self, items=[], limit=15, cols=1, rows=1, depth=1, ratiosX=None, ratiosY=None, createView=None, **args):
        super().__init__(**args)
        self.items = items if type(items) == list else [items]
        self.limit = limit
        self.totalRows = rows
        self.totalCols = cols
        self.totalDepth = depth
        self.totalLength = self.totalRows * self.totalCols * self.totalDepth

        if createView != None:
            self.createView = createView

        self.ci, self.cj, self.ck = 0, 0, 0
        self.rows = min(self.totalRows, self.limit)
        self.cols = min(self.totalCols, self.limit)
        self.depth = min(self.totalDepth, self.limit)
        self.length = self.rows * self.cols * self.depth
        self.canHold = True
        self.isHidingViews = False

        self.containers = []
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    index = self.index(i, j, k)
                    view = self.createView(self, self.totalIndex(i, j, k))
                    container = Container(view=view, container=self,
                                          ratioX=1.0 / self.cols if ratiosX == None else ratiosX[index],
                                          ratioY=1.0 / self.rows if ratiosY == None else ratiosY[index])
                    self.containers.append(container)

    def createView(self, table, index):
        return self.items[index]

    def index(self, i, j, k):
        return j + self.cols * (i + self.rows * k) if i >= 0 and i < self.rows and j >= 0 and j < self.cols and k >= 0 and k < self.depth else None

    def totalIndex(self, i, j, k):
        return j + self.totalCols * (i + self.totalRows * k) if i >= 0 and i < self.totalRows and j >= 0 and j < self.totalCols and k >= 0 and k < self.totalDepth else None

    def display(self):
        if not self.isHidden:
            super().display()
            for container in self.containers:
                container.display()

    def clicked(self, x, y):
        if self.isClicked(x, y):
            if self.isDraggable:
                return self
            for i in range(len(self.containers) - 1, -1, -1):
                view = self.getView(i)
                if view != None:
                    clickedObj = view.clicked(x, y)
                    if clickedObj != None:
                        return clickedObj
            return self

    def updateDown(self):
        super().updateDown()
        for container in self.containers:
            container.updateDown()

    def getLeafOrder(self):
        for container in self.containers:
            yield from container.getLeafOrder()
        yield self

    def findRootOrder(self, q):
        yield self
        for container in self.containers:
            q.append(container)

    def updateAlignment(self, isX=False, isY=False):
        index, y = (0, 0.0)
        for i in range(self.rows):
            x = 0.0
            for j in range(self.cols):
                for k in range(self.depth):
                    c = self.containers[index]
                    dx, dy = hp.calcAlignment(x=x, y=y, dw=self.getWidth() - c.getWidth(),
                                              dh=self.getHeight() - c.getHeight(), isX=isX, isY=isY)
                    c.setAlignment(dx=dx, dy=dy)
                    index += 1
                x += self.containers[index - 1].getWidth()
            y += self.containers[index - 1].getHeight()

    def findContainerLengths(self, total, count, minLengthKey, ratioKey, maintainOrder=True):
        self.containers.sort(key=lambda x: x.get(minLengthKey), reverse=True)
        lengths = []
        totalRatio = 1.0
        for container in self.containers:
            length, minLength = (total * container.get(ratioKey) / totalRatio, container.get(minLengthKey))
            if minLength > length:
                total -= minLength
                count -= 1
                totalRatio -= container.get(ratioKey)
                lengths.append((container, minLength))
            else:
                lengths.append((container, length))
        if maintainOrder:
            self.containers.sort(key=lambda x: x.id)
        return lengths

    def shift(self, dx=0, dy=0, dz=0):
        if self.ci < dy or self.ci + self.rows > self.totalRows + dy:
            dy = 0
        if dx == 0 and dy == 0 and dz == 0:
            return

        cx, cy, cz = ((0 if dx <= 0 else self.cols - 1),
                      (0 if dy <= 0 else self.rows - 1),
                      (0 if dz <= 0 else self.depth - 1))
        ddx, ddy, ddz = -1 if dx > 0 else 1, -1 if dy > 0 else 1, -1 if dz > 0 else 1

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    x, y, z = cx + j * ddx, cy + i * ddy, cz + k * ddz
                    a, b = self.index(y - dy, x - dx, z - dz), self.index(y, x, z)

                    if a != None and 2 * y != dy + 1:
                        x2, y2, z2 = x + self.cj, y + self.ci, z + self.ck
                        index = self.totalIndex(y2 - 2 * dy, x2 - 2 * dx, z2 - 2 * dz)
                        newView = self.createView(self, index)
                        if newView != None:
                            if b != None:
                                view = self.getView(a)
                                if view != None:
                                    view.setContainer(self.containers[b])
                                # self.arr[b] = self.arr[a]
                            if y > 1 and y < 14:
                                newView.setContainer(self.containers[a])

                    #     if a != b:
                    #         self.arr[a] = 0
                    # elif b != None:
                    #     self.arr[b] = 0
        self.ci -= dy
        self.cj -= dx
        self.ck -= dz

    def getEmptyContainer(self, x, y):
        for view in self.getRootOrder():
            if view.isContainer() and view.view == None and view.isWithin(x, y):
                return view

    def canDragView(self, view, container):
        return False

    def getView(self, key):
        return self.containers[key].view

    def addView(self, item, ratioX=None, ratioY=None):
        if ratioX == None:
            ratioX = 1 / self.cols
        if ratioY == None:
            ratioY = 1 / self.rows

        totalRatioX, totalRatioY = 1.0 - ratioX, 1.0 - ratioY
        for container in self.containers:
            container.updateRatios(ratioX=totalRatioX * container.ratioX, ratioY=totalRatioY * container.ratioY)

        self.items.append(item)
        if len(self.items) <= self.limit:
            self.containers.append(Container(view=self.createView(self, len(self.items) - 1), container=self, ratioX=ratioX, ratioY=ratioY))

    def popView(self):
        self.items.pop()
        return self.containers.pop().view

    def __len__(self):
        return len(self.containers)

    def __str__(self, indent=""):
        return "Stack:{}[\n  {}{}\n{}]".format(self.getID(), indent, ",\n  {}".format(indent).join([x.__str__(indent=indent + "  ") for x in self.containers]), indent)

    def __getitem__(self, key):
        return self.containers[key]


class HStack(Stack):

    def __init__(self, items=[], ratios=None, **args):
        super().__init__(items=items, cols=len(items) if type(items) == list else 1, ratiosX=ratios, **args)

    def updateFrame(self):
        super().updateFrame()
        lengths = self.findContainerLengths(total=self.getWidth(), count=len(self), minLengthKey="minWidth", ratioKey="ratioX")
        for container, length in lengths:
            container.setSize(width=length, height=self.getHeight())
        self.updateAlignment(isX=True)

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        containerMinWidth = 0.0
        for container in self.containers:
            containerMinWidth += container.minWidth
            self.minHeight = max(self.minHeight, container.minHeight)
        self.minWidth = max(self.minWidth, containerMinWidth)

    def addView(self, item, ratio=None):
        self.cols += 1
        super().addView(item, ratioX=ratio)

    def popView(self):
        self.cols -= 1
        return super().popView()

    def __str__(self, indent=""):
        return "H{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class VStack(Stack):

    def __init__(self, items=[], ratios=None, **args):
        super().__init__(items=items, rows=len(items) if type(items) == list else 1, ratiosY=ratios, **args)

    def updateFrame(self):
        super().updateFrame()
        lengths = self.findContainerLengths(total=self.getHeight(), count=len(self), minLengthKey="minHeight", ratioKey="ratioY")
        for container, length in lengths:
            container.setSize(width=self.getWidth(), height=length)
        self.updateAlignment(isY=True)

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        containerMinHeight = 0.0
        for container in self.containers:
            self.minWidth = max(self.minWidth, container.minWidth)
            containerMinHeight += container.minHeight
        self.minHeight = max(self.minHeight, containerMinHeight)

    def addView(self, item, ratio=None):
        self.rows += 1
        super().addView(item, ratioY=ratio)

    def popView(self):
        self.rows -= 1
        return super().popView()

    def __str__(self, indent=""):
        return "V{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class Grid(Stack):

    def __init__(self, items=[], rows=1, cols=1, **args):
        super().__init__(items=items, rows=rows, cols=cols, **args)

    def updateFrame(self):
        super().updateFrame()
        widths = self.findContainerLengths(total=self.getWidth(), count=self.cols, minLengthKey="minWidth", ratioKey="ratioX", maintainOrder=False)
        heights = self.findContainerLengths(total=self.getHeight(), count=self.rows, minLengthKey="minHeight", ratioKey="ratioY")
        widths.sort(key=lambda x: x[0].id)
        heights.sort(key=lambda x: x[0].id)

        for i in range(len(self)):
            container, width = widths[i]
            _, height = heights[i]
            container.setSize(width=width, height=height)
        self.updateAlignment(isX=True, isY=True)

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        containerMinWidth, containerMinHeight = (0.0, 0.0)
        for container in self.containers:
            containerMinWidth += container.minWidth
            containerMinHeight += container.minHeight
        self.minWidth = max(self.minWidth, containerMinWidth)
        self.minHeight = max(self.minHeight, containerMinHeight)

    def addView(self, item):
        "Not implemented!!"
        pass

    def popView(self):
        "Not implemented!!"
        pass

    def __str__(self, indent=""):
        return "Grid:{}".format(self.getID())

    def __getitem__(self, key):
        return super().__getitem__(key)


class ZStack(Stack):

    def __init__(self, items=[], **args):
        super().__init__(items=items, depth=len(items) if type(items) == list else 1, **args)

    def updateFrame(self):
        super().updateFrame()
        for container in self.containers:
            container.setSize(width=self.getWidth(), height=self.getHeight())

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        for container in self.containers:
            self.minWidth = max(self.minWidth, container.minWidth)
            self.minHeight = max(self.minHeight, container.minHeight)

    def addView(self, item):
        self.depth += 1
        super().addView(item)

    def popView(self):
        self.depth -= 1
        return super().popView()

    def scrollUp(self):
        pass

    def scrollDown(self):
        pass

    def display(self):
        super().display()
        # if self.findKey("codeStack"):
        #     pg.draw.rect(g, Color.red, (self.x, self.y, self.getWidth(), self.getHeight()))

    def __str__(self, indent=""):
        return "Z{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class Button(ZStack):
    def __init__(self, items, run=None, isOn=True, setViewMethod=None, clickHoldTime=0.5, **args):
        super().__init__(items=items, **args)
        self.run = run
        self.setViewMethod = setViewMethod
        self.clickedTime = None
        self.clickHoldTime = clickHoldTime
        self.isOn = None
        self.setOn(isOn=isOn)

    def display(self):
        if not self.isHidden:
            if self.clickedTime and time() - self.clickedTime > self.clickHoldTime:
                self.clickedTime = None
                self.setOn(None)  # force update
            super().display()

    def clicked(self, x, y):
        if self.clickedTime == None and self.isClicked(x, y):
            self.isOn = not self.isOn
            self.clickedTime = time()
            self.runSetView()
            if self.run != None:
                self.run(self)
            if self.isDraggable:
                return self
            return False  # Nothing else can be clicked above in the hiearchy(False != None)

    def setOn(self, isOn):
        if self.isOn != isOn:
            if isOn != None:
                self.isOn = isOn
            self.runSetView()
            # (self.viewBackup if self.isOn or self.toggleView == True else self.toggleView).setContainer(container=self, allowButtonUpdate=False)

    def runSetView(self):
        if self.setViewMethod != None:
            self.setViewMethod(self)
            stack = self.getParentStack()
            if stack != None:
                stack.updateAll()

    def isAlt(self):
        return self.clickedTime != None

    def __str__(self, indent=""):
        return "Button:{}{}".format(self.getID(), super().__str__(indent=indent))
