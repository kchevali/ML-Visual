import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp

FPS = 50
width = 1050
height = 700
title = "Teaching APP"
isDebug = True

# Cannot import from graphics
backgroundColor = ((50, 50, 50))


def update():
    global mouseDebug
    # print("START")
    # background = pg.Surface(self.size)
    # background.fill(hp.red)
    # INPUT======================================
    mousePos = pg.mouse.get_pos()
    if isDebug:
        mouseDebug.keyDown("text").setFont(text=str(mousePos))
        mouseDebug.updateAll()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            return False
        if event.type == pg.MOUSEBUTTONDOWN:
            if page != None:
                page.clicked(*mousePos)

    # DRAW=======================================
    g.fill(backgroundColor)
    # g.blit(background, (0, 0))
    if page != None:
        page.display()

    # -----======================================
    pg.display.update()
    clock.tick(FPS)
    return True


def setPage(p):
    global page, mouseDebug
    page = p
    if page != None:
        if isDebug:
            from page import createMouseDebug
            mouseDebug = createMouseDebug()
            page.addView(mouseDebug)
        page.setSize(width=width, height=height)
        page.updateAll()


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
