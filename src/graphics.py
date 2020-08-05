from gui import g
from time import time
import helper as hp
import pygame as pg

# ===========================================================
# FRAMES
# ===========================================================


class Frame:

    def __init__(self, dx, dy, border=10):
        self._width = 0.0
        self._height = 0.0
        self.minWidth = 0.0
        self.minHeight = 0.0
        self.dx = dx
        self.dy = dy
        self.border = border
        self.isWidthLocked = True
        self.isHeightLocked = True
        self.isContainer = False
        self.setContainer(None)

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

    def setBorder(self, border):
        self.border = border

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

    def clicked(self, x, y):
        pass

    def setDim(self, width=None, height=None):
        self.updateFrame()

    def display(self):
        pass
        # pg.draw.rect(g, hp.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)


class ResizableFrame(Frame):
    def __init__(self, dx, dy, border, lockedWidth, lockedHeight):
        super().__init__(dx=dx, dy=dy, border=border)
        self.lockWidth(lockedWidth)
        self.lockHeight(lockedHeight)

    def setDim(self, width=None, height=None):
        if width != None and not self.isWidthLocked:
            self._width = width
        if height != None and not self.isHeightLocked:
            self._height = height
        self.updateMinimumDimensions()
        self.updateFrame()

    def lockWidth(self, lockedWidth):
        self.isWidthLocked = lockedWidth != None
        if self.isWidthLocked:
            self._width = lockedWidth
            self.minWidth = lockedWidth

    def lockHeight(self, lockedHeight):
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


class Color(ResizableFrame):

    def __init__(self, color, dx=0.0, dy=0.0, border=0.0, lockedWidth=None, lockedHeight=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.color = color

    def display(self):
        super().display()
        pg.draw.rect(g, self.color, (self.x, self.y, self.getWidth(), self.getHeight()))


class Label(Frame):
    def __init__(self, text, fontName="Comic Sans MS", fontSize=32, autoFontSize=False, color=hp.white, dx=0.0, dy=0.0):
        super().__init__(dx, dy)
        self.autoFontSize = autoFontSize
        self.setFont(text=text, fontName=fontName, fontSize=fontSize, color=color)

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

    def __init__(self, view, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None, container=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.view = view
        self._addView(self.view)
        self.setContainer(container)
        self.isContainer = True

    def _addView(self, view):
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
        pg.draw.rect(g, hp.green, (self.x, self.y, self.getWidth(), self.getHeight()), 2)
        self._displayView(self.view)

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
        self.clickHoldTime = 2.0

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
                self.run()


# ===========================================================
# STACKS
# ===========================================================

class Stack(ResizableFrame):

    # init args have default values for ZStack()
    def __init__(self, views, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        self.containers = [Container(view=view, container=self) for view in views]

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

    def __len__(self):
        return len(self.containers)


class HStack(Stack):

    def __init__(self, views, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(views=views, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        ratio = 0.0 if len(self) <= 1 else (2.0 / (len(self) - 1))
        index = 0
        for container in self.containers:
            container.setAlignment(dx=ratio * index - 1.0)
            index += 1

    def updateFrame(self):
        super().updateFrame()
        # assert self._width >= self.minWidth, "HStack Error: width constraint compromised"
        self.setDimView(containers=self.containers.copy(), totalLength=self._width)

    def setDimView(self, containers, totalLength):
        if len(containers) == 0 or totalLength <= 0.0:
            return

        length = totalLength / len(containers)
        isFail = False
        cp = containers.copy()
        for i in range(len(containers)):
            container = cp[i]
            if container.minWidth > length:
                totalLength -= container.minWidth
                containers.setDim(width=container.minWidth, height=self.getHeight())
                isFail = True
        if isFail:
            self.setDimView(containers, totalLength)
            return
        for container in containers:
            container.setDim(width=length, height=self.getHeight())
            totalLength -= length
        assert totalLength == 0, "HStack Error: width constraint not met"


class VStack(Stack):

    def __init__(self, views, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None):
        super().__init__(views=views, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)
        ratio = 0.0 if len(self) <= 1 else (2.0 / (len(self) - 1))
        index = 0
        for container in self.containers:
            container.setAlignment(dy=ratio * index - 1.0)
            index += 1

    def updateFrame(self):
        super().updateFrame()
        # TODO fix height checking
        # assert self._height >= self.minHeight, "VStack Error: height constraint compromised - Stack Height: {} Min: {}".format(self._height, self.minHeight)
        self.setDimView(containers=self.containers.copy(), totalLength=self._height)

    def setDimView(self, containers, totalLength):
        if len(containers) == 0 or totalLength <= 0.0:
            return
        length = totalLength / len(containers)
        isFail = False
        cp = containers.copy()
        for i in range(len(containers)):
            container = cp[i]
            if container.minHeight > length or container.isHeightLocked:
                containers.pop(i)
                totalLength -= container.minHeight
                container.setDim(width=self.getWidth(), height=container.minHeight)
                isFail = True
        if isFail:
            self.setDimView(containers, totalLength)
            return
        for container in containers:
            container.setDim(width=self.getWidth(), height=length)
            totalLength -= length
        assert totalLength == 0, "VStack Error: height constraint not met - {}".format(totalLength)


class ZStack(Stack):

    # def __init__(self, views, dx=0.0, dy=0.0, border=10):
    #     super().__init__(views=views, dx=dx, dy=dy, border=border)
        # not needed - since default is 0.0
        # for container in self.containers:
        #     container.setAlignment(dx=0.0, dy=0.0)

    def updateFrame(self):
        super().updateFrame()
        for container in self.containers:
            container.setDim(width=self.getWidth(), height=self.getHeight())
