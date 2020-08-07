import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp

FPS = 50
width = 1000
height = 700
title = "Teaching APP"


def update():
    # background = pg.Surface(self.size)
    # background.fill(hp.red)
    # INPUT======================================
    for event in pg.event.get():
        if event.type == pg.QUIT:
            return False
        if event.type == pg.MOUSEBUTTONDOWN:
            if page != None:
                page.clicked(*pg.mouse.get_pos())

    # DRAW=======================================
    g.fill(hp.backgroundColor)
    # g.blit(background, (0, 0))
    if page != None:
        page.display()

    # -----======================================
    pg.display.update()
    clock.tick(FPS)
    return True


def setPage(p):
    global page
    page = p
    if page != None:
        page.setDim(width=width, height=height)


def close(self):
    pg.quit()


pg.init()
pg.font.init()
# hp.initFontSizer('Comic Sans MS', 1, 128)
size = (width, height)
pg.display.set_caption(title)
clock = pg.time.Clock()
g = pg.display.set_mode(size)
setPage(None)


if __name__ == '__main__':
    hp.clear()
    print("RUNNING GUI Main")
