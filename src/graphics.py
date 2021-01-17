from gui import g
from time import time
import helper as hp
import pygame as pg
from pygame.mixer import Sound
import pygame.freetype
from colorsys import hsv_to_rgb
from collections import deque

import numpy as np

# ===========================================================
# FRAMES
# ===========================================================


class Frame:

    idCounter = 0

    def __init__(self, name="", tag=0, keywords="", dx=0.0, dy=0.0, offsetX=0.0, offsetY=0.0, isHidden=False, hideContainer=False, hideAllContainers=False, isDraggable=False, isDisabled=False, container=None, **kwargs):
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
        # self.setup = []
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

    # def addInstruction(self, method, args):
    #     self.setup.append((method, args))

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
        return None
        # if self.isClicked(x, y):
        #     return None

    def setSize(self, width=None, height=None):
        "Empty Method"

    def hoverMouse(self, x, y):
        "Empty Method"

    def update(self):
        pass

    def getSize(self):
        return (self.getWidth(), self.getHeight())

    def scrollUp(self):
        pass

    def scrollDown(self):
        pass

    # Does not depend on isDraggable - this is asking if the container will accept the view
    # See gui.py
    def canDragView(self, view, container):
        return False

    def display(self):
        "Empty Method"
        # if not self.isHidden:
        #     pass
        # pg.draw.rect(g, Color.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)

    def get(self, key):
        return self.__dict__[key]


class ResizableFrame(Frame):
    def __init__(self, lockedWidth=None, lockedHeight=None, border=0, **kwargs):
        super().__init__(**kwargs)
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

    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
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

    def __init__(self, strokeColor=None, strokeWidth=None, **kwargs):
        super().__init__(**kwargs)
        self.strokeColor = strokeColor
        self.strokeWidth = None if self.strokeColor == None else strokeWidth


class Rect(Shape):

    def __init__(self, color, cornerRadius=0, border=10, **kwargs):
        super().__init__(color=color, border=border, **kwargs)
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


class Points(Ellipse):
    # pts should be between -1 and 1
    def __init__(self, pts=[], color=None, isConnected=False, maxPts=0, ptSize=5, isCircle=True, **kwargs):
        super().__init__(color=color, **kwargs)
        self.setPts(pts)  # [-1,1]
        self.isConnected = isConnected
        self.maxPts = maxPts
        self.ptSize = ptSize
        self.isCircle = isCircle
        self.displayPts = []  # pixel coordinates

    def updateFrame(self):
        super().updateFrame()
        self.displayPts = [self.map(pt) for pt in self.pts]
        # print(self.displayPts)

    def addPt(self, pt):
        if len(self.pts) < self.maxPts:
            self.pts.append(pt)
            self.displayPts.append(self.map(pt))
        else:
            self.reset()

    def setPts(self, pts):
        self.pts = pts
        self.displayPts = [self.map(pt) for pt in self.pts]

    def map(self, pt):
        return (
            hp.map(pt[0], -1, 1, self.x, self.x + self.getWidth(), clamp=False),
            hp.map(pt[1], -1, 1, self.y, self.y + self.getHeight(), clamp=False),
            pt[2]
        )

    def display(self):
        if not self.isHidden:
            if not self.isConnected:
                if self.isCircle:
                    for i in range(len(self.displayPts)):
                        pg.draw.ellipse(g, self.displayPts[i][2], (self.displayPts[i][0], self.displayPts[i][1], self.ptSize, self.ptSize))
                else:
                    for i in range(len(self.displayPts)):
                        pg.draw.rect(g, self.displayPts[i][2], (self.displayPts[i][0], self.displayPts[i][1], self.ptSize, self.ptSize))
            else:
                for i in range(1, len(self.displayPts)):
                    pg.draw.line(g, self.color, tuple(self.displayPts[i - 1][0:2]), tuple(self.displayPts[i][0:2]), width=self.ptSize)

    def setColor(self, index, color):
        self.pts[index] = (self.pts[index][0], self.pts[index][1], color)
        self.displayPts[index] = (self.displayPts[index][0], self.displayPts[index][1], color)

    def reset(self):
        self.pts = []
        self.displayPts = []


class Image(ResizableFrame):

    def __init__(self, imageName, angle=0.0, **kwargs):
        super().__init__(**kwargs)
        self.imageName = imageName
        self.angle = angle

        self.surface = pg.transform.rotate(pg.image.load("assets/images/" + imageName), self.angle)
        self.setSize(*self.surface.get_size())
        if self.isWidthLocked or self.isHeightLocked:
            self.surface = pg.transform.scale(self.surface, (self.getWidth(), self.getHeight()))

    def display(self):
        if not self.isHidden:
            super().display()
            g.blit(self.surface, (self.x, self.y, self.getWidth(), self.getHeight()))

    def __str__(self, indent=""):
        return "Image:'{}'".format(self.imageName)


class Label(Frame):
    # border has no effect on
    fontCache = {}

    def __init__(self, text, fontName="Sans", fontSize=24, color=Color.white, autoFontSize=False, isVertical=False, **kwargs):
        super().__init__(**kwargs)
        self.autoFontSize = autoFontSize
        self.setFont(text=text, fontName=fontName, fontSize=fontSize, color=color, isVertical=isVertical)

    def setFont(self, text=None, fontName=None, fontSize=None, color=None, isVertical=None):
        if text != None:
            self.text = [text] if type(text) != list else text
        if fontName != None:
            self.fontName = fontName
        if fontSize != None:  # and not self.autoFontSize:
            self.fontSize = fontSize
        if color != None:
            self.color = color
        if isVertical != None:
            self.isVertical = isVertical

        fontKey = self.fontName + str(self.fontSize) + str(self.isVertical)
        if fontKey in Label.fontCache:
            self.font = Label.fontCache[fontKey]
        else:
            self.font = pg.freetype.SysFont(self.fontName, self.fontSize)
            Label.fontCache[fontKey] = self.font

        self.font.vertical = self.isVertical
        self.updateFrame()  # -- most current/ removed since causing extra updates
        # however, better calls are still needed

        # self.render(self.text)
        # self.threeDots = False

    def render(self, lines):
        self.rects = []
        width, height = 0.0, 0.0
        x, y = self.pos
        for i, line in enumerate(lines):
            rect = self.font.get_rect(line)
            rect.topleft = (x, y)
            self.rects.append(rect)
            width = max(width, rect.width)
            height += rect.height
            y += rect.height
        self._setSize(width, height)

        #     surface = self.font.render(line, True, self.color)
        #     # lineWidth, lineHeight = self.font.size(line)
        #     lineWidth, lineHeight = surface.get_size()
        #     width = max(width, lineWidth)
        #     height += lineHeight
        #     surfaces.append(surface)
        #     heights.append(lineHeight + heights[-1])

        # self.surface = pg.Surface((width, height), pg.SRCALPHA)

        # for i in range(len(surfaces)):
        #     self.surface.blit(surfaces[i], (0, heights[i]))
        # if self.angle != 0.0:
        #     self.surface = pg.transform.rotate(self.surface, self.angle)
        # self._setSize(*self.surface.get_size())

    def updateFrame(self):
        super().updateFrame()
        self.render(self.text)
        # if not self.threeDots and not self.doesFit():
        #     self.threeDots = True
        #     self.render("...")

    def display(self):
        if not self.isHidden:
            super().display()
            for i in range(len(self.text)):
                self.font.render_to(g, self.rects[i].topleft, self.text[i], self.color, size=self.fontSize)
            # g.blit(self.surface, self.pos)
            # for i, surface in enumerate(self.surfaces):
            #     g.blit(surface, (self.x, self.y + self.fontSize * i))

    def __str__(self, indent=""):
        return "Label:'{}'".format(self.text[0])

# ===========================================================
# CONTAINERS
# ===========================================================

# Why containers? A container is a fixed frame relative to its view. The view can scale and position itself freely within the container


class Container(ResizableFrame):

    def __init__(self, view=None, ratioX=1.0, ratioY=1.0, showEmpty=False, **kwargs):
        super().__init__(**kwargs)
        self.view = view
        self.canHold = True
        self.ratioX = ratioX
        self.ratioY = ratioY
        self.showEmpty = showEmpty
        if view != None:
            view.setContainer(container=self)

    def updateFrame(self):
        super().updateFrame()
        if self.view != None:
            self.view.setSize(width=self.getWidth() - 2 * self.view.border, height=self.getHeight() - 2 * self.view.border)

    def display(self):
        if not self.isHidden:
            if (self.view == None and self.showEmpty) or (self.view != None and not self.view.hideContainer and (self.container == None or not self.container.hideAllContainers)):
                pg.draw.rect(g, Color.darkGray, (self.x, self.y, self.getWidth(), self.getHeight()), 2)
            super().display()
            if self.view != None:
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

    def updateRatios(self, ratioX=None, ratioY=None):
        if ratioX != None:
            self.ratioX = ratioX
        if ratioY != None:
            self.ratioY = ratioY

    def setContainer(self, container):
        self.container = container

    def delink(self):
        raise Exception("Cannot delink a container")


# ===========================================================
# STACKS
# ===========================================================


class Stack(ResizableFrame):

    # init args have default values for ZStack()
    def __init__(self, items=[], limit=15, cols=1, rows=1, depth=1, ratiosX=[], ratiosY=[], containerArgs=[], createCellViewMethod=None, hoverEnabled=False, **kwargs):
        super().__init__(**kwargs)
        self.buildStack(items=items, limit=limit, cols=cols, rows=rows, depth=depth, ratiosX=ratiosX, ratiosY=ratiosY,
                        containerArgs=containerArgs, createCellViewMethod=createCellViewMethod, hoverEnabled=hoverEnabled)
        self.canHold = True
        self.isHidingViews = False

    def buildStack(self, items=None, limit=None, cols=None, rows=None, depth=None, ratiosX=None, ratiosY=None, containerArgs=None, createCellViewMethod=None, hoverEnabled=None, **kwargs):
        if not items is None:
            self.items = items if type(items) == list or type(items) == np.ndarray else [items]
        if limit != None:
            self.limit = limit
        if rows != None:
            self.totalRows = rows
        if cols != None:
            self.totalCols = cols
        if depth != None:
            self.totalDepth = depth
        if ratiosX != None:
            self.ratiosX = ratiosX
        if ratiosY != None:
            self.ratiosY = ratiosY
        if containerArgs != None:
            self.containerArgs = containerArgs
        if createCellViewMethod != None:
            self.createCellView = createCellViewMethod
        if hoverEnabled != None:
            self.hoverEnabled = hoverEnabled

        self.ci, self.cj, self.ck = 0, 0, 0
        self.rows = min(self.totalRows, self.limit)
        self.cols = min(self.totalCols, self.limit)
        self.depth = min(self.totalDepth, self.limit)
        self.totalLength = self.totalRows * self.totalCols * self.totalDepth
        self.length = self.rows * self.cols * self.depth

        # self.selectedRow = None
        # self.selectedCol = None
        # self.selectedDepth = None

        self.containers = []
        # print("total:", self.rows, self.cols, self.depth, self.length)
        # print("Items:", self.items)
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(self.depth):
                    index = self.index(i, j, k)
                    # print("Index:", i, j, k, index)
                    view = self.createCellView(self, self.totalIndex(i, j, k))
                    container = Container(view=view, container=self,
                                          ratioX=1.0 / self.cols if len(self.ratiosX) == 0 else self.ratiosX[index],
                                          ratioY=1.0 / self.rows if len(self.ratiosY) == 0 else self.ratiosY[index],
                                          **self.containerArgs[index] if index < len(self.containerArgs) else {})
                    self.containers.append(container)

    def createCellView(self, selfObj, index):
        return self.items[index]

    # def isSelected(self, i, j, k):
    #     return self.selectedRow == i or self.selectedCol == j or self.selectedDepth == k

    def index(self, i, j, k):
        return j + self.cols * (i + self.rows * k) if i >= 0 and i < self.rows and j >= 0 and j < self.cols and k >= 0 and k < self.depth else None

    def totalIndex(self, i, j, k):
        return j + self.totalCols * (i + self.totalRows * k) if i >= 0 and i < self.totalRows and j >= 0 and j < self.totalCols and k >= 0 and k < self.totalDepth else None

    def coord(self, index):
        div = index // self.cols
        return div % self.rows, index % self.cols, div // self.rows

    def totalCoord(self, index):
        div = index // self.totalCols
        return div % self.totalRows, index % self.totalCols, div // self.totalRows

    def display(self):
        if not self.isHidden:
            super().display()
            for container in self.containers:
                container.display()

    def clicked(self, x, y):
        if self.isClicked(x, y):
            if self.isDraggable:
                return self
            for view in self.getRevViews():
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
                    dx, dy = hp.calcAlignment(x=x, y=y, dw=self.getWidth() - c.getWidth(), dh=self.getHeight() - c.getHeight(), isX=isX, isY=isY)
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
                        newView = self.createCellView(self, index)
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

    def getEmptyContainers(self, x, y):
        for view in self.getRootOrder():
            if view.isContainer() and view.view == None and view.isWithin(x, y):
                yield view

    def canDragView(self, view, container):
        if(super().canDragView(view, container)):
            return True
        for c in self.containers:
            if c.view != None and c.view.canDragView(view, container):
                return True
        return False

    def draggedView(self, view):
        for c in self.containers:
            if c.view != None:
                c.view.draggedView(view=view)

    def hoverMouse(self, x, y):
        for c in self.containers:
            if c.view != None:
                c.view.hoverMouse(x, y)

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
            self.containers.append(Container(view=self.createCellView(self, len(self.items) - 1), container=self, ratioX=ratioX, ratioY=ratioY))

    def popView(self):
        self.items.pop()
        return self.containers.pop().view

    def getViews(self):
        for c in self.containers:
            if c != None:
                yield c.view

    def getRevViews(self):
        for c in reversed(self.containers):
            if c.view != None:
                yield c.view

    def clear(self):
        for _ in range(len(self.items)):
            self.popView()

    def peekView(self):
        return self.containers[-1].view

    def scrollUp(self):
        for view in self.getViews():
            if view != None:
                view.scrollUp()

    def scrollDown(self):
        for view in self.getViews():
            if view != None:
                view.scrollDown()

    def update(self):  # used in linear regression example page
        for c in self.containers:
            if c.view != None:
                c.view.update()

    def __len__(self):
        return len(self.containers)

    def __str__(self, indent=""):
        return "Stack:{}[\n  {}{}\n{}]".format(self.getID(), indent, ",\n  {}".format(indent).join([x.__str__(indent=indent + "  ") for x in self.containers]), indent)

    def __getitem__(self, key):
        return self.containers[key]


class HStack(Stack):

    def __init__(self, items=[], ratios=[], **kwargs):
        super().__init__(items=items, cols=len(items) if type(items) == list else 1, ratiosX=ratios, **kwargs)

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

    def __init__(self, items=[], ratios=[], **kwargs):
        super().__init__(items=items, rows=len(items) if type(items) == list else 1, ratiosY=ratios, **kwargs)

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

    def __init__(self, items=[], rows=1, cols=1, table=None, **kwargs):
        if table == None:
            super().__init__(items=items, rows=rows, cols=cols, **kwargs)
        else:
            super().__init__(items=table.flatten(), rows=table.rowCount + 1, cols=table.colCount + 1, **kwargs)

    def build(self, items=[], rows=1, cols=1, table=None):
        if table == None:
            self.buildStack(items=items, rows=rows, cols=cols)
        else:
            self.buildStack(items=table.flatten(), rows=table.rowCount + 1, cols=table.colCount + 1)

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

    def __init__(self, items=[], **kwargs):
        super().__init__(items=items, depth=len(items) if type(items) == list else 1, **kwargs)

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

    def addAllViews(self, *args):
        for item in args:
            self.addView(item)

    def popView(self):
        self.depth -= 1
        return super().popView()

    # def display(self):
    #     super().display()
        # if self.findKey("codeStack"):
        #     pg.draw.rect(g, Color.red, (self.x, self.y, self.getWidth(), self.getHeight()))

    def __str__(self, indent=""):
        return "Z{}".format(super().__str__(indent=indent))

    def __getitem__(self, key):
        return super().__getitem__(key)


class Button(ZStack):
    def __init__(self, items, run=None, isOn=True, setViewMethod=None, clickHoldTime=0.5, soundName="click", volume=0.02, **kwargs):
        super().__init__(items=items, **kwargs)
        self.run = run
        self.setViewMethod = setViewMethod
        self.clickedTime = None
        self.clickHoldTime = clickHoldTime
        self.isOn = None
        self.setOn(isOn=isOn)
        self.lastClickX = 0
        self.lastClickY = 0
        self.setSoundName(soundName=soundName, volume=volume)

    def setSoundName(self, soundName, volume=0.02):
        if soundName != None:
            self.sound = Sound(hp.resourcePath("assets/audio/" + soundName + ".wav"))
            self.sound.set_volume(volume)
        else:
            self.sound = None

    def display(self):
        if not self.isHidden:
            # pg.draw.rect(g, Color.red, (self.x, self.y, self.getWidth(), self.getHeight()))
            if self.clickedTime and time() - self.clickedTime > self.clickHoldTime:
                self.clickedTime = None
                self.setOn(None)  # force update
            super().display()

    def clicked(self, x, y):
        if self.clickedTime == None and self.isClicked(x, y):
            self.lastClickX = x
            self.lastClickY = y
            self.isOn = not self.isOn
            self.clickedTime = time()
            self.runSetView()
            if self.sound != None:
                Sound.play(self.sound)
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
        return "Button-{}".format(super().__str__(indent=indent))


if __name__ == '__main__':
    pass
    # def createCellView(selfObj, index):
    #     return Rect(color=Color.blue)

    # from table import Table
    # table = Table(filePath="examples/decisionTree/movie")
    # grid = Grid(table=table)
    # index = 5
    # coords = grid.coord(index)
    # index2 = grid.index(*coords)
    # print("Rows:", grid.rows, "Cols:", grid.cols, "Depth:", grid.depth)
    # count = 0
    # # print("Match:", index, index2, coords)
    # for i in range(grid.length):
    #     if index == index2:
    #         count += 1
    # print("PASS" if count == grid.length else str(100 * count / grid.length))
