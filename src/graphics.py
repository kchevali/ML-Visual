from gui import g
from time import time
import helper as hp
import pygame as pg
import pygame.gfxdraw as pgx

# ===========================================================
# FRAMES
# ===========================================================


class Frame:

    idCounter = 0

    def __init__(self, dx, dy, container=None):
        self._width = 0.0
        self._height = 0.0
        self.minWidth = 0.0
        self.minHeight = 0.0
        self.dx = dx
        self.dy = dy
        self.container = container
        self.border = 0.0  # unused by fixed frames
        self.id = Frame.idCounter
        Frame.idCounter += 1

        # CONSTS CANNOT CHANGE
        self.isWidthLocked = True
        self.isHeightLocked = True
        self.isContainer = False

        # TEMP METHOD FOR X,Y
        self._setXY(x=0.0, y=0.0)
        # self.updateFrame()

    def _setXY(self, x, y):
        self.x = x
        self.y = y
        self.pos = (self.x, self.y)

    def setAlignment(self, dx=None, dy=None):
        if dx != None:
            self.dx = dx
        if dy != None:
            self.dy = dy
        self.updateFrame()

    def setContainer(self, container):
        self.container = container

    def getWidth(self):
        assert self._width >= self.minWidth or not self.isWidthLocked, "Width contraints are compromised"
        return max(self._width, self.minWidth)

    def getHeight(self):
        assert self._height >= self.minHeight or not self.isHeightLocked, "Height contraints are compromised"
        return max(self._height, self.minHeight)

    def updateFrame(self):
        if self.container != None:
            self._setXY(x=self.container.x + (self.dx + 1.0) * (self.container.getWidth() - self.getWidth()) / 2.0,
                        y=self.container.y + (self.dy + 1.0) * (self.container.getHeight() - self.getHeight()) / 2.0)

    def isWithin(self, x, y):
        return x >= self.x and x <= self.x + self.getWidth() and y >= self.y and y <= self.y + self.getHeight()

    def updateMinimumDimensions(self):
        self.minWidth = self._width
        self.minHeight = self._height
        # print("Frame UMD:", self.minWidth, self.minHeight)
        if self.container != None and not self.isContainer:
            self.container.updateMinimumDimensions()

    def _setDim(self, width, height):
        self._width = width
        self._height = height
        self.updateMinimumDimensions()

    def replace(self, view):
        if self.container != None:
            self.container.view = view
            view.container = self.container
            self.container = None
            self.updateMinimumDimensions()
            view.container.updateFrame()
        return view

    def clicked(self, x, y):
        pass

    def setDim(self, width=None, height=None):
        self.updateFrame()

    def display(self):
        pass
        # pg.draw.rect(g, hp.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)

    def __getitem__(self, key):
        return self.__dict__[key]


class ResizableFrame(Frame):
    def __init__(self, dx, dy, border, lockedWidth, lockedHeight, container=None):
        super().__init__(dx=dx, dy=dy, container=container)
        self._lockWidth(lockedWidth)
        self._lockHeight(lockedHeight)
        self.border = border

    def setDim(self, width=None, height=None):
        if width != None and not self.isWidthLocked:
            self._width = width
        if height != None and not self.isHeightLocked:
            self._height = height
        # TODO figure out when this should get called
        # self.updateMinimumDimensions()
        self.updateFrame()

    def _lockWidth(self, lockedWidth):
        self.isWidthLocked = lockedWidth != None
        if self.isWidthLocked:
            self._width = lockedWidth
            self.minWidth = lockedWidth

    def _lockHeight(self, lockedHeight):
        self.isHeightLocked = lockedHeight != None
        if self.isHeightLocked:
            self._height = lockedHeight
            self.minHeight = lockedHeight

    def updateMinimumDimensions(self):
        self.minWidth = self._width if self.isWidthLocked else 0.0
        self.minHeight = self._height if self.isHeightLocked else 0.0
        # print("Resize UMD:", self.minWidth, self.minHeight)
        if self.container != None and not self.isContainer:
            self.container.updateMinimumDimensions()

    def setBorder(self, border):
        self.border = border


class Color(ResizableFrame):

    def __init__(self, color, dx=0.0, dy=0.0, border=0.0, lockedWidth=None, lockedHeight=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.color = color

    def display(self):
        super().display()
        pg.draw.rect(g, self.color, (self.x, self.y, self.getWidth(), self.getHeight()))


class Shape(Color):

    def __init__(self, color=None, strokeColor=None, strokeWidth=0, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(color=color, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.strokeColor = strokeColor
        self.strokeWidth = strokeWidth


class Rect(Shape):

    def __init__(self, color=None, strokeColor=None, strokeWidth=0, cornerRadius=0, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(color=color, strokeColor=strokeColor, strokeWidth=strokeWidth, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.cornerRadius = cornerRadius
        if self.strokeColor == None:
            self.strokeWidth = None

    def display(self):
        # if self.color != None:
        #     hp.draw_bordered_rounded_rect(surface=g, color=self.color, rect=(self.x, self.y, self.getWidth(), self.getHeight()), width=0, border_radius=self.cornerRadius)
        # if self.storkeColor != None and self.strokeWidth > 0:
        hp.draw_bordered_rounded_rect(g, rect=(self.x, self.y, self.getWidth(), self.getHeight()), color=self.color,
                                      border_color=self.strokeColor, corner_radius=self.cornerRadius, border_thickness=self.strokeWidth)


class Ellipse(Shape):

    def display(self):
        x, y, w, h = tuple(map(int, [self.x, self.y, self.getWidth() / 2, self.getHeight() / 2]))
        x += w
        y += h
        if self.color != None:
            pgx.filled_ellipse(g, x, y, w, h, self.color)
        if self.strokeColor != None and self.strokeWidth > 0:
            pgx.aaellipse(g, x, y, w, h, self.strokeColor)


class Label(Frame):
    # border has no effect on
    def __init__(self, text, fontName="Comic Sans MS", fontSize=32, autoFontSize=False, color=hp.white, dx=0.0, dy=0.0):
        super().__init__(dx=dx, dy=dy)
        self.autoFontSize = autoFontSize
        self.setFont(text, fontName, fontSize, color)
        # self.text = text
        # self.fontName = fontName
        # self.fontSize = fontSize
        # self.color = color
        # self.font = pg.font.SysFont(self.fontName, self.fontSize)
        # self.width, self.height = self.font.size(self.text)
        # self.surface = self.font.render(self.text, True, self.color)

    def setFont(self, text=None, fontName=None, fontSize=None, color=None):
        if text != None:
            self.text = text
        if fontName != None:
            self.fontName = fontName
        if fontSize != None and not self.autoFontSize:
            self.fontSize = fontSize
        if color != None:
            self.color = color
        self.font = pg.font.SysFont(self.fontName, self.fontSize)
        self._setDim(*self.font.size(self.text))
        self.surface = self.font.render(self.text, True, self.color)
        self.updateFrame()

    def display(self):
        super().display()
        g.blit(self.surface, self.pos)

# ===========================================================
# CONTAINERS
# ===========================================================


class Container(ResizableFrame):

    def __init__(self, view, ratioX=1.0, ratioY=1.0, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None, container=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight, container=container)
        self.view = view
        self._addView(self.view)
        self.isContainer = True
        self.ratioX = ratioX  # ratio within a stack
        self.ratioY = ratioY

    def _addView(self, view):
        if view != None:
            view.container = self
            self._updateFrameView(view)

    def _updateFrameView(self, view):
        if view != None:
            view.setDim(width=self.getWidth() - 2 * view.border, height=self.getHeight() - 2 * view.border)
            # view.updateFrame()  # not needed - called in setDim

    def _displayView(self, view):
        if view != None:
            view.display()

    def _clickedView(self, view, x, y):
        if view != None:
            view.clicked(x, y)

    def updateFrame(self):
        super().updateFrame()
        self._updateFrameView(self.view)

    def display(self):
        super().display()
        self._displayView(self.view)
        pg.draw.rect(g, hp.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)

    def clicked(self, x, y):
        self._clickedView(self.view, x, y)

    def updateMinimumDimensions(self):
        super().updateMinimumDimensions()
        if self.view != None:
            # self.view.updateMinimumDimensions()#pretty sure not neede
            self.minWidth = max(self.minWidth, self.view.minWidth)
            self.minHeight = max(self.minHeight, self.view.minHeight)

        if self.container != None:
            self.container.updateMinimumDimensions()


class Button(Container):
    def __init__(self, view=None, altView=None, run=None, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(view=view, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.altView = altView
        self._addView(self.altView)
        self.run = run

        self.clickedTime = None
        self.clickHoldTime = 0.5

    def updateFrame(self):
        super().updateFrame()
        self._updateFrameView(self.altView)

    def display(self):
        if self.clickedTime != None:
            if time() - self.clickedTime <= self.clickHoldTime:
                self._displayView(self.altView)
                return
            self.clickedTime = None

        super().display()  # only if not display alternate view

    def clicked(self, x, y):
        if self.isWithin(x, y):
            # doesn't make sense to click altView - leave super() only here
            super().clicked(x, y)
            self.clickedTime = time()
            if self.run != None:
                self.run(self)


# ===========================================================
# STACKS
# ===========================================================

class Stack(ResizableFrame):

    # init args have default values for ZStack()
    def __init__(self, views, ratiosX=None, ratiosY=None, cols=1, rows=1, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.cols = cols
        self.rows = rows
        self.containers = [Container(view=views[i], container=self, border=0, ratioX=1.0 / self.cols if ratiosX == None else ratiosX[i],
                                     ratioY=1.0 / self.rows if ratiosY == None else ratiosY[i]) for i in range(len(views))]

        fracY, fracX = (0.0 if self.rows <= 1 else (2.0 / (self.rows - 1)),
                        0.0 if self.cols <= 1 else (2.0 / (self.cols - 1)))
        index = 0
        for i in range(self.rows):
            for j in range(self.cols):
                self.containers[index].setAlignment(dx=(fracX * j - 1.0), dy=(fracY * i - 1.0))
                index += 1

    def display(self):
        super().display()
        for container in self.containers:
            container.display()

    def clicked(self, x, y):
        if(self.isWithin(x, y)):
            for container in self.containers:
                container.clicked(x, y)

    def updateMinimumDimensions(self):
        super().updateMinimumDimensions()
        containerMinWidth, containerMinHeight = (0.0, 0.0)
        for container in self.containers:
            # self.view.updateMinimumDimensions()#pretty sure not neede
            containerMinWidth += container.minWidth
            containerMinHeight += container.minHeight
        self.minWidth = max(self.minWidth, containerMinWidth)
        self.minHeight = max(self.minHeight, containerMinHeight)
        # print("Stack DIM:", self._width, self._height, "MIN:", self.minWidth, self.minHeight)
        if self.container != None:
            self.container.updateMinimumDimensions()

    def findContainerLengths(self, total, count, minLengthKey, ratioKey):
        self.containers.sort(key=lambda x: x[minLengthKey], reverse=True)
        lengths = []
        totalRatio = 1.0
        for container in self.containers:
            length, minLength = (total * container[ratioKey] / totalRatio, container[minLengthKey])
            if minLength > length:
                total -= minLength
                count -= 1
                totalRatio -= container.ratio[ratioKey]
                lengths.append((container, 0.0))
            else:
                lengths.append((container, length))
        return lengths

    def __len__(self):
        return len(self.containers)


class HStack(Stack):

    def __init__(self, views, ratios=None, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        #super().__init__() is important
        super().__init__(views=views, cols=len(views), ratiosX=ratios, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)

    def updateFrame(self):
        super().updateFrame()
        # assert self._width >= self.minWidth, "HStack Error: width constraint compromised"
        lengths = self.findContainerLengths(total=self.getWidth(), count=len(self), minLengthKey="minWidth", ratioKey="ratioX")
        for container, length in lengths:
            container.setDim(width=length, height=self.getHeight())


class VStack(Stack):

    def __init__(self, views, ratios=None, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        #super().__init__() is important
        super().__init__(views=views, rows=len(views), ratiosY=ratios, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)

    def updateFrame(self):
        super().updateFrame()
        # TODO fix height checking
        # assert self._height >= self.minHeight, "VStack Error: height constraint compromised - Stack Height: {} Min: {}".format(self._height, self.minHeight)
        lengths = self.findContainerLengths(total=self.getHeight(), count=len(self), minLengthKey="minHeight", ratioKey="ratioY")
        for container, length in lengths:
            container.setDim(width=self.getWidth(), height=length)


class ZStack(Stack):

    def __init__(self, views, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        # needed to restrict init args when creating ZStack
        super().__init__(views=views, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)

    def updateFrame(self):
        super().updateFrame()
        for container in self.containers:
            container.setDim(width=self.getWidth(), height=self.getHeight())


class Grid(Stack):

    def __init__(self, views, cols, rows, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(views=views, cols=cols, rows=rows, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)

    def updateFrame(self):
        super().updateFrame()
        widths = self.findContainerLengths(total=self.getWidth(), count=self.cols, minLengthKey="minWidth", ratioKey="ratioX")
        heights = self.findContainerLengths(total=self.getHeight(), count=self.rows, minLengthKey="minHeight", ratioKey="ratioY")
        widths.sort(key=lambda x: x[0].id)
        heights.sort(key=lambda x: x[0].id)

        for i in range(len(self)):
            container, width = widths[i]
            _, height = heights[i]
            container.setDim(width=width, height=height)
