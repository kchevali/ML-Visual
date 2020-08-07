import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
from graphics import *
from gui import g
import pandas as pd
import numpy as np


class Table(Grid):

    # input should be (headers + data) or (cols + rows)
    def __init__(self, data=None, filePath=None, dx=0.0, dy=0.0, border=10, lockedWidth=None, lockedHeight=None, fontName="Comic Sans MS", fontSize=32, autoFontSize=False, fontColor=hp.white):
        self.data = data if data != None else pd.read_csv(filePath)
        headerViews = [Label(str(label), fontName=fontName, fontSize=fontSize, autoFontSize=autoFontSize, color=fontColor)for label, _ in self.data.items()]
        mainViews = [Label(str(row[label]), fontName=fontName, fontSize=fontSize, autoFontSize=autoFontSize, color=fontColor) for index, row in self.data.iterrows() for label, _ in self.data.items()]
        super().__init__(views=headerViews + mainViews,
                         cols=len(self.data.columns), rows=len(self.data) + 1, dx=dx, dy=dy, border=border, lockedWidth=lockedWidth, lockedHeight=lockedHeight)


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
