import pygame as pg
import pygame.freetype
# from pygame import gfxdraw as pgx
import helper as hp
from event import Event

if not 'g' in globals():
    g = None
    windowWidth, windowHeight = 0, 0


def createGUI():
    global windowWidth, windowHeight, g
    windowWidth = 1050
    windowHeight = 700
    pg.init()
    # pg.font.init()
    pg.freetype.init()
    g = pg.display.set_mode((windowWidth, windowHeight), pg.RESIZABLE)


def runGUI(page):
    global windowWidth, windowHeight, g
    print("Loading...", end="\r")
    title = "Teaching APP"
    pg.display.set_caption(title)
    clock = pg.time.Clock()

    FPS = 50
    isDebug = False
    dragObj = None

    # Cannot import from graphics
    backgroundColor = ((50, 50, 50))
    page.setSize(width=windowWidth, height=windowHeight)
    page.updateAll()
    print("Loading Complete!")


    # INPUT======================================
    isRun = True
    while isRun:
        page.update()
        mouse = pg.mouse.get_pos()
        page.mouseEvent(Event(), mouse)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                isRun = False
            elif event.type == pg.VIDEORESIZE:
                page.setSize(width=event.w, height=event.h)
                page.updateAll()
                g = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
            elif event.type == pg.MOUSEBUTTONDOWN or event.type == pg.MOUSEBUTTONUP:
                page.mouseEvent(event, mouse)

        # DRAW=======================================
        g.fill(backgroundColor)
        # g.blit(background, (0, 0))
        page.display()
        # -----======================================
        pg.display.update()
        clock.tick(FPS)
    pg.quit()
    pg.font.quit()
    pg.freetype.quit()


if __name__ == '__main__':
    hp.clear()
    print("RUNNING GUI Main")
    pg.init()
    pg.font.init()


"""

"""
