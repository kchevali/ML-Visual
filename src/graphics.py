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

    def __init__(self, name="", tag=0, keywords="", dx=0.0, dy=0.0, offsetX=0.0, offsetY=0.0, isHidden=False, container=None):
        # INPUTS
        self.name = name
        self.tag = tag
        self.keywords = keywords
        self.dx = dx
        self.dy = dy
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.isHidden = isHidden
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
        self.isDisabled = False
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
        if dx:
            self.dx = dx
        if dy:
            self.dy = dy

    def addInstruction(self, method, args):
        self.setup.append((method, args))

    def delink(self, allowButtonUpdate=True):
        if self.container:
            self.container.view = None
            if self.container.isButton() and allowButtonUpdate:
                self.container.viewBackup = None
            self.container = None

    def setContainer(self, container, allowButtonUpdate=True):
        self.delink(allowButtonUpdate=allowButtonUpdate)
        if container != None:
            if container.view:
                container.view.delink(allowButtonUpdate=allowButtonUpdate)
            self.container = container
            self.container.view = self
            if self.container.isButton() and allowButtonUpdate:
                self.container.viewBackup = self
            # setup = self.setup
            # self.setup = []
            # for method, args in setup:
            #     method(*args)

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

    # need to call when updating x/y/w/h or containers x/y/w/h
    def updateFrame(self):
        if self.container:
            self._setXY(x=self.container.x + (self.dx + 1.0) * (self.container.getWidth() - self.getWidth()) / 2.0,
                        y=self.container.y + (self.dy + 1.0) * (self.container.getHeight() - self.getHeight()) / 2.0)

    def updateAll(self):
        views = self.getLeafOrder()
        for view in views:
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
        q = deque()
        self.findLeafOrder(q)
        return q

    def findLeafOrder(self, q):
        q.appendleft(self)

    def getRootOrder(self):
        q = [self]
        index = 0
        while index < len(q):
            q[index].findRootOrder(q)
            index += 1
        return q

    def findRootOrder(self, q):
        pass

    def updateDown(self):
        self.updateFrame()

    def findKey(self, keywords):
        return self if keywords in self.keywords else None

    def keyUp(self, keywords, excludeSelf=False):
        if not excludeSelf:
            obj = self.findKey(keywords)
            if obj:
                return obj
        if self.container:
            return self.container.keyUp(keywords)

    def keyDown(self, keywords, excludeSelf=False):
        for obj in self.getRootOrder():
            if obj.findKey(keywords) and (not excludeSelf or self != obj):
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
        if width and not self.isWidthLocked:
            self._width = width
        if height and not self.isHeightLocked:
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

    def __init__(self, cornerRadius=0, border=10, **args):
        super().__init__(border=border, **args)
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
    def __init__(self, text, fontName="Comic Sans MS", fontSize=32, color=Color.white, autoFontSize=False, **args):
        super().__init__(**args)
        self.autoFontSize = autoFontSize
        self.setFont(text=text, fontName=fontName, fontSize=fontSize, color=color)

    def setFont(self, text=None, fontName=None, fontSize=None, color=None):
        if text != None:
            self.text = text
        if fontName:
            self.fontName = fontName
        if fontSize:  # and not self.autoFontSize:
            self.fontSize = fontSize
        if color:
            self.color = color
        self.font = pg.font.SysFont(self.fontName, self.fontSize)
        self.render(self.text)
        self.threeDots = False

    def render(self, text):
        self.surface = self.font.render(text, True, self.color)
        self._setSize(*self.font.size(text))

    def updateFrame(self):
        super().updateFrame()
        if not self.threeDots and not self.doesFit():
            self.threeDots = True
            self.render("...")

    def display(self):
        if not self.isHidden:
            super().display()
            g.blit(self.surface, self.pos)

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
        if view:
            view.setContainer(container=self)

    def updateFrame(self):
        super().updateFrame()
        if self.view:
            self.view.setSize(width=self.getWidth() - 2 * self.view.border, height=self.getHeight() - 2 * self.view.border)

    def display(self):
        if not self.isHidden:
            pg.draw.rect(g, Color.darkGray, (self.x, self.y, self.getWidth(), self.getHeight()), 2)
            super().display()
            if self.view:
                self.view.display()

    def clicked(self, x, y):
        if self.isClicked(x, y):
            clickedObj = self.view.clicked(x, y) if self.view else None
            return clickedObj if clickedObj else self

    def updateDown(self):
        super().updateDown()
        if self.view:
            self.view.updateDown()

    def findLeafOrder(self, q):
        if self.view:
            self.view.findLeafOrder(q)
        q.append(self)

    def findRootOrder(self, q):
        if self.view:
            q.append(self.view)

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        if self.view:
            self.minWidth = max(self.minWidth, self.view.minWidth)
            self.minHeight = max(self.minHeight, self.view.minHeight)

    def __str__(self, indent=""):
        return "<{}>".format(self.view.__str__(indent=indent) if self.view else None)


class Container(Holder):

    def __init__(self, ratioX=1.0, ratioY=1.0, **args):
        super().__init__(**args)
        self.ratioX = ratioX
        self.ratioY = ratioY

    def setContainer(self, container):
        self.container = container

    def delink(self):
        raise Exception("Cannot delink a container")


class Button(Holder):
    def __init__(self, view, altView=None, toggleView=None, run=None, isOn=True, **args):
        super().__init__(view=view, **args)
        self.altView = altView
        self.toggleView = toggleView
        self.run = run
        self.isOn = None
        self.viewBackup = self.view
        self.clickedTime = None
        self.clickHoldTime = 0.5
        self.setOn(isOn=isOn)

    def display(self):
        if not self.isHidden:
            if self.clickedTime and time() - self.clickedTime > self.clickHoldTime:
                self.setOn(None)  # force update
                self.getParentStack().updateAll()

                # (self.viewBackup if self.isOn else self.toggleView).setContainer(container=self)

                self.clickedTime = None
            super().display()
            # pg.draw.rect(g, Color.red, (self.x, self.y, self.getWidth(), self.getHeight()), 2)

    def clicked(self, x, y):
        if self.clickedTime == None:
            if self.isClicked(x, y):
                if self.toggleView:
                    self.isOn = not self.isOn
                if self.altView:
                    self.altView.setContainer(container=self, allowButtonUpdate=False)
                    self.clickedTime = time()
                    self.getParentStack().updateAll()
                else:
                    self.clickedTime = 1.0
                if self.run:
                    self.run(self)

    def setOn(self, isOn):
        if self.isOn != isOn:
            if isOn != None:
                self.isOn = isOn
            (self.viewBackup if self.isOn or self.toggleView == True else self.toggleView).setContainer(container=self, allowButtonUpdate=False)

    def __str__(self, indent=""):
        return "Button:{}{}".format(self.getID(), super().__str__(indent=indent))


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
        if nextView:
            nextDisjoint = nextView.container
            nextView.setContainer(container=self.view.container)
            self.view.setContainer(container=self.disjoint)
            self.disjoint = nextDisjoint
            self.view = nextView


# ===========================================================
# STACKS
# ===========================================================

class Stack(ResizableFrame):

    # init args have default values for ZStack()
    def __init__(self, items=[], limit=15, cols=1, rows=1, depth=1, ratiosX=None, ratiosY=None, createView=None, **args):
        super().__init__(**args)
        self.items = items
        self.limit = limit
        self.totalRows = rows
        self.totalCols = cols
        self.totalDepth = depth
        self.totalLength = self.totalRows * self.totalCols * self.totalDepth

        if createView:
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
                    self.containers.append(Container(view=self.createView(self, self.totalIndex(i, j, k)), container=self,
                                                     ratioX=1.0 / self.cols if ratiosX == None else ratiosX[index],
                                                     ratioY=1.0 / self.rows if ratiosY == None else ratiosY[index]))

    def createView(self, sender, index):
        return sender.items[index]

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
            for i in range(len(self.containers) - 1, -1, -1):
                view = self.getView(i)
                if view:
                    clickedObj = view.clicked(x, y)
                    if clickedObj:
                        return clickedObj
            return self

    def updateDown(self):
        super().updateDown()
        for container in self.containers:
            container.updateDown()

    def findLeafOrder(self, q):
        for container in self.containers:
            container.findLeafOrder(q)
        q.append(self)

    def findRootOrder(self, q):
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
        cx, cy, cz = ((0 if dx <= 0 else self.cols - 1),
                      (0 if dy <= 0 else self.rows - 1),
                      (0 if dz <= 0 else self.depth - 1))
        ddx, ddy, ddz = -1 if dx > 0 else 1, -1 if dy > 0 else 1, -1 if dz > 0 else 1

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    x, y, z = cx + j * ddx, cy + i * ddy, cz + k * ddz
                    a, b = self.index(y - dy, x - dx, z - dz), self.index(y, x, z)

                    if a != None:
                        x2, y2, z2 = x + self.cj, y + self.ci, z + self.ck
                        index = self.totalIndex(y2 - 2 * dy, x2 - 2 * dx, z2 - 2 * dz)
                        # if a != b and ((i + dy) >= self.rows or (j + dx >= self.cols) or k + dz >= self.depth):
                        newView = self.createView(self, index)
                        if newView:
                            if b != None and y > dy and y > 0:
                                view = self.getView(a)
                                if view:
                                    view.setContainer(self.containers[b])
                                # self.arr[b] = self.arr[a]
                            newView.setContainer(self.containers[a])

                    #     if a != b:
                    #         self.arr[a] = 0
                    # elif b != None:
                    #     self.arr[b] = 0
        self.ci -= dy
        self.cj -= dx
        self.ck -= dz

    def getView(self, key):
        return self.containers[key].view

    def __len__(self):
        return len(self.containers)

    def __str__(self, indent=""):
        return "Stack:{}[\n  {}{}\n{}]".format(self.getID(), indent, ",\n  {}".format(indent).join([x.__str__(indent=indent + "  ") for x in self.containers]), indent)

    def __getitem__(self, key):
        return self.containers[key]


class HStack(Stack):

    def __init__(self, items=[], ratios=None, **args):
        super().__init__(items=items, cols=len(items), ratiosX=ratios, **args)

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

    def __str__(self, indent=""):
        return "H{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class VStack(Stack):

    def __init__(self, items=[], ratios=None, **args):
        super().__init__(items=items, rows=len(items), ratiosY=ratios, **args)

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

    def __str__(self, indent=""):
        return "V{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class ZStack(Stack):

    def __init__(self, items=[], **args):
        super().__init__(items=items, depth=len(items), **args)

    def updateFrame(self):
        super().updateFrame()
        for container in self.containers:
            container.setSize(width=self.getWidth(), height=self.getHeight())

    def updateMinimumSizes(self):
        super().updateMinimumSizes()
        for container in self.containers:
            self.minWidth = max(self.minWidth, container.minWidth)
            self.minHeight = max(self.minHeight, container.minHeight)

    def addView(self, view):
        self.containers.append(Container(view=view, container=self))
        self.depth += 1

    def popView(self):
        self.depth -= 1
        return self.containers.pop().view

    def __str__(self, indent=""):
        return "Z{}".format(super().__str__(indent=indent))

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

    def __str__(self, indent=""):
        return "Grid:{}".format(self.getID())

    def __getitem__(self, key):
        return super().__getitem__(key)
