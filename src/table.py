import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
from frame import Frame
import pandas as pd
import numpy as np


class Table(Frame):

    # input should be (headers + data) or (cols + rows)
    def __init__(self, data):  # headers=None, data=None
        super().__init__(isWidthFixed=False, isHeightFixed=False)
        self.data = data
        self.cols = len(data.columns)
        self.rows = len(data)
        # print("Created Table -- Cols:", self.cols, "Rows:", self.rows)
        # print(self.data)
        # if headers != None and data != None:
        # self.headers = np.array(headers)
        # self.data = pd.DataFrame(data=data, columns=headers)
        # self.cols = len(self.data)
        # self.rows = len(self.data[0]) if self.cols > 0 else 0
        # else:
        #     self.cols = cols
        #     self.rows = rows
        #     # self.headers = ["- -" for _ in range(self.cols)]
        #     self.data = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        # print("Cols: {} Rows: {}".format(self.cols, self.rows))

    def display(self):
        if self.g == None:
            return
        super().display()

        cellWidth = self.width / self.cols
        cellHeight = self.height / (self.rows + 1)
        textBorder = 5

        for i in range(self.rows + 2):
            y = self.y + cellHeight * i
            pg.draw.line(self.g, hp.white, (self.x, y), (self.x + self.width, y), 3)

        for i in range(self.cols + 1):
            x = self.x + cellWidth * i
            pg.draw.line(self.g, hp.white, (x, self.y), (x, self.y + self.height), 3)

        x = self.x
        for label, content in self.data.items():

            text, textWidth, textHeight = hp.getFont(label, cellWidth, cellHeight, hp.red)
            pos = hp.findPosition(textWidth, textHeight, x, self.y, cellWidth, cellHeight)
            self.g.blit(text, pos)
            y = self.y + cellHeight

            for data in content:
                # print("Item:", item)
                # for i in range(len(self.data)):
                #     for j in range(len(self.data[i])):
                # data = self.data[i][j] if self.data[i][j] != None else 0
                text, textWidth, textHeight = hp.getFont(str(data), cellWidth, cellHeight, hp.red)
                pos = hp.findPosition(textWidth, textHeight, x, y, cellWidth, cellHeight)
                self.g.blit(text, pos)
                y += cellHeight
            x += cellWidth

    @staticmethod
    def readCSV(filePath):
        return Table(data=pd.read_csv(filePath))


if __name__ == '__main__':
    hp.clear()
    print("Running TABLE MAIN")
    table = Table.readCSV("examples/decisionTree.csv")

    from gui import GUI
    gui = GUI("Table MAIN", width=1000, height=600)
    window = gui.newWindow()
    window.setDrawObject(table)
    while gui.update():
        pass
    gui.close()

    # table = Table.readCSV("examples/decisionTree.csv")
    # print(table.header)
