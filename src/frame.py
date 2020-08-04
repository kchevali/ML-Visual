import helper as hp
import pygame as pg


class Frame:

    def __init__(self, x=0.0, y=0.0, width=0.0, height=0.0, isWidthFixed=False, isHeightFixed=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.isWidthFixed = isWidthFixed
        self.isHeightFixed = isHeightFixed
        self.g = None

    def updateFrame(self, x=None, y=None, width=None, height=None):
        if x != None:
            self.x = x
        if y != None:
            self.y = y

        if width != None and not self.isWidthFixed:
            self.width = width
        if height != None and not self.isHeightFixed:
            self.height = height

    def findPosition(self, obj):
        return hp.findPosition(width=obj.width, height=obj.height, containerX=self.x, containerY=self.y, containerWidth=self.width, containerHeight=self.height, dx=self.dx, dy=self.dy)

    def isWithin(self, x, y):
        return x >= self.x and x <= self.x + self.width and y >= self.y and y <= self.y + self.height

    def clicked(self, x, y):
        pass

    def display(self):
        if self.g == None:
            return
        # pg.draw.rect(self.g, hp.green, (self.x, self.y, self.width, self.height), 2)
