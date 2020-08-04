import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp
from frame import Frame
from window import Window


class GUI:

    instance = None
    FPS = 1

    def __init__(self, title, width, height):

        if GUI.instance != None:
            GUI.instance.close()
            GUI.instance = self

        self.width = width
        self.height = height
        self.title = title
        self.dim = (width, height)
        self.windows = []
        self.window = None

        pg.init()
        hp.initFontSizer('Comic Sans MS', 1, 128)
        pg.display.set_caption(title)
        self.g = pg.display.set_mode(self.dim)
        self.clock = pg.time.Clock()

    def update(self):
        # background = pg.Surface(self.dim)
        # background.fill(hp.red)
        # INPUT======================================
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.MOUSEBUTTONDOWN:
                mouseX, mouseY = pg.mouse.get_pos()
                if self.window != None:
                    self.window.checkClicked(mouseX, mouseY)

        # DRAW=======================================
        self.g.fill(hp.backgroundColor)
        # self.g.blit(background, (0, 0))
        if self.window != None:
            self.window.display()

        # -----======================================
        pg.display.update()
        self.clock.tick(GUI.FPS)
        return True

    def newWindow(self):
        window = Window(name="main", width=self.width, height=self.height, g=self.g)
        self.windows.append(window)
        if self.window == None:
            self.window = window
        return window

    def selectWindow(self, key):
        self.window = self.windows[key]

    def setWindow(self, window):
        self.window = window

    def close(self):
        pg.quit()
        GUI.instance = None


if __name__ == '__main__':
    hp.clear()
    print("RUNNING GUI Main")
    gui = GUI(title="GUI Main", width=500, height=500)
    # while gui.update():
    #     pass
    # gui.close()
    # print(pg.font.get_fonts())
    for i in range(1, 32):
        font = pg.font.SysFont('Comic Sans MS', i)
        text_width, text_height = font.size("txt")
        print("{}. WIDTH: {} HEIGHT: {}".format(i, text_width, text_height))
