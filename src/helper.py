import os
import pygame as pg
from time import time
from random import randint
from os import listdir
from os.path import isfile, join

white = (255, 255, 255)
red = ((255, 0, 0))
green = (0, 255, 0)
backgroundColor = ((50, 50, 50))


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


fonts = []
fontInc = 3


def initFontSizer(fontName, minSize, maxSize):
    global fonts
    pg.font.init()
    fonts = [pg.font.SysFont(fontName, i) for i in range(minSize, maxSize + 1, fontInc)]
    # print("Font Ready")


def getFont(txt, targetWidth, targetHeight, color):
    # print("Target Width:", targetWidth, "Height:", targetHeight)
    font = searchFont(txt, targetWidth, targetHeight, a=0, b=len(fonts) - 1)
    width, height = font.size(txt)
    # print("Final Width:", width, "Height:", height)
    return font.render(txt, True, color), width, height


def searchFont(txt, targetWidth, targetHeight, a, b):

    index = (a + b) // 2
    font = fonts[index]
    width, height = font.size(txt)
    # print("A:", a, "B:", b, "W:", width, "H:", height)

    if a >= b:
        return font if index == 0 or (width <= targetHeight and height <= targetWidth) else fonts[index - 1]
    if width <= targetWidth and height <= targetHeight:
        return searchFont(txt, targetWidth, targetHeight, index + 1, b)
    return searchFont(txt, targetWidth, targetHeight, a, index - 1)


def findPosition(width, height, containerX, containerY, containerWidth, containerHeight, dx=0.0, dy=0.0):
    return containerX + (dx + 1.0) * (containerWidth - width) / 2, containerY + (dy + 1.0) * (containerHeight - height) / 2


def randomString():
    return str(randint(0, 1 << 31))


def getFiles(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


if __name__ == '__main__':
    clear()
    print("Running HELPER MAIN")
    initFontSizer('Comic Sans MS', 1, 32)
    _ = getFont("Hello", 80, 20, white)
