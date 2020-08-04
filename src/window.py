from frame import Frame
import pygame as pg
import helper as hp
import json


class Window(Frame):

    def __init__(self, name, x=0.0, y=0.0, width=0.0, height=0.0, dx=0.0, dy=0.0, ratio=1.0, g=None, parent=None):
        super().__init__(x=x, y=y, width=width, height=height, isWidthFixed=False, isHeightFixed=False)
        self.name = name
        self.dx = dx
        self.dy = dy
        self.obj = None
        self.windows = {}
        self.ratio = ratio
        self.g = g
        self.parent = parent
        self.isVertical = None
        self.root = self.parent.root if self.parent != None else self
        # print("Created Window --- Name:", name, "Ratio:", ratio)

    def updateFrame(self, x=None, y=None, width=None, height=None):
        super().updateFrame(x, y, width, height)
        self.updateSubWindows()
        self.updateObject()
        # print("Update Window: X: {} Y: {} W: {} H: {}".format(x, y, width, height))

    def updateObject(self):
        if self.obj != None:
            self.obj.updateFrame(width=self.width, height=self.height)
            objX, objY = self.findPosition(self.obj)
            self.obj.updateFrame(x=objX, y=objY)

    def setObject(self, obj=None, dx=None, dy=None):
        if dx != None:
            self.dx = dx
        if dy != None:
            self.dy = dy
        if obj != None:
            self.obj = obj
            self.obj.g = self.g
            self.obj.parent = self

        self.updateObject()

    def splitWindow(self, names, isVertical, ratios=None):
        self.isVertical = isVertical
        count = len(names)

        for i in range(count):
            name = names[i] if names != None and names[i] != None else hp.randomString()
            while name in self.windows:
                name = hp.randomString()

            # print("Split Window - Name:", name, "Ratio:", ratios[i] if ratios != None else 1.0 / count)
            window = Window(
                name=name,
                ratio=float(ratios[i]) if ratios != None else 1.0 / count,
                g=self.g,
                parent=self
            )
            self.windows[name] = window
            if self.root != self:
                self.root.windows[name] = window

        self.updateSubWindows()

    # def splitWindow2D(self, count):
    #     self.splitWindow(count, isVertical=False)
    #     for i in range(count):
    #         self.windows[i].splitWindow(count, isVertical=True)

    def splitFromDict(self, data):
        # print("Split from Dict:", self.name, self.name in data)
        if self.name in data:
            entry = data[self.name]
            # print("Entry:", entry, "Names:", entry["names"], type(entry["names"]))
            self.splitWindow(names=entry["names"], isVertical=entry["isVertical"], ratios=entry["ratios"])
            keys = list(self.windows.keys())
            for key in keys:
                self.windows[key].splitFromDict(data)

    def splitFromFile(self, filePath):
        with open(filePath) as f:
            self.splitFromDict(json.load(f))

    def setRatios(self, ratios):
        for name, ratio in ratios.items():
            self.windows[name].ratio = ratio
        self.updateSubWindows()

    def updateSubWindows(self):
        if len(self.windows) > 0:
            isX, isY = (0, 1) if self.isVertical else (1, 0)
            x, y = (self.x, self.y)
            for name, window in self.windows.items():
                width = self.width * (1.0 if self.isVertical else window.ratio)
                height = self.height * (window.ratio if self.isVertical else 1.0)
                window.updateFrame(x=x, y=y, width=width, height=height)
                x += width * isX
                y += height * isY

    def display(self):
        # super().display()
        for name, window in self.windows.items():
            window.display()
        if self.obj != None:
            self.obj.display()
        pg.draw.rect(self.g, hp.red, (self.x, self.y, self.width, self.height), 1)

    def checkClicked(self, x, y):
        if self.isWithin(x, y):
            for name, window in self.windows.items():
                window.checkClicked(x, y)
            if self.obj != None and self.obj.isWithin(x, y):
                self.obj.clicked(x, y)

    def __getitem__(self, key):
        try:
            return self.windows[key]
        except:
            print("Error: Cannot find page '{}'".format(key))

    @staticmethod
    def createFromDict(data, width, height, g):
        window = Window(name=data["name"], ratio=data["ratio"], width=width, height=height, g=g)
        window.isVertical = data["isVertical"]
        for windowData in data["subpage"]:
            window.split
