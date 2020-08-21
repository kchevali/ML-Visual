import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
from graphics import *
from gui import g
import pandas as pd
import numpy as np


class Table(Grid):

    # input should be (headers + data) or (cols + rows)
    def __init__(self, data=None, param=None, filePath=None, limit=10, fontName="Comic Sans MS", fontSize=32, autoFontSize=False, fontColor=Color.white, **args):
        self.param = hp.loadJSON(filePath + ".json") if not param else param
        self.targetName = self.param['target']
        self.indexCol = self.param['index']
        self.colNames = self.param['columns']

        self.data = data if data is not None else pd.read_csv(filePath + ".csv")  # , index_col=self.indexCol
        self.data = self.data[[self.targetName] + self.colNames]  # move target to front and limit to given columns
        self.targetCol = self.data[self.targetName]
        self.loc = self.data.loc
        self.classes = self.targetCol.unique()
        self.classCount = self.targetCol.nunique()

        cols = min(len(self.data.columns), limit)
        rows = min(len(self.data) + 1, limit)  # not include header row
        self.dataRows = min(len(self.data), limit)
        print("Created Table | Cols:", cols, "Rows:", rows)
        headerViews, mainViews, countX, countY = ([], [], 0, 0)
        for column in self.data:
            headerViews.append(self.createCellView(column, fontName=fontName, fontSize=fontSize, autoFontSize=autoFontSize, fontColor=fontColor, rectColor=Color.steelBlue))

        for index, row in self.data.iterrows():
            if countY >= limit - 1:  # sub 1 for header row
                break
            countX = 0
            for label, _ in self.data.items():
                if countX >= limit:
                    break
                mainViews.append(self.createCellView(row[label], fontName=fontName, fontSize=fontSize, autoFontSize=autoFontSize, fontColor=fontColor, rectColor=Color.lightSteelBlue))
                countX += 1
            countY += 1
        super().__init__(items=headerViews + mainViews, cols=cols, rows=rows, **args)
        self.selectedColumns = [False for _ in range(self.cols)]
        # self.selectColumn(True, 2)
        # print("Adding Overlay")
        # self.addInstruction(self.addOverlay, ([Rect(color=hp.blue, cornerRadius=10)], 1))

    def createCellView(self, obj, fontName, fontSize, autoFontSize, fontColor, rectColor):
        return ZStack(items=[
            Rect(color=rectColor, border=3, cornerRadius=5, keywords="rect"),
            Label(text=str(obj), fontName=fontName, fontSize=fontSize, autoFontSize=autoFontSize, color=fontColor, keywords="label")
        ])

    def selectColumn(self, value, index):
        self.selectedColumns[index] = value
        # self.getView(index).getView(0).color = Color.steelBlue if value else Color.steelBlue
        for i in range(1, self.rows):
            self.getView(i * self.cols + index).keyDown("rect").color = Color.steelBlue if value else Color.lightSteelBlue

    def iterrows(self):
        return self.data.iterrows()

    def __str__(self, indent=""):
        return "Table:{}".format(self.id)

    def __getitem__(self, key):
        return self.data[key]


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
