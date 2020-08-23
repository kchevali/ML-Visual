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
    mouseX, mouseY = pg.mouse.get_pos()
    if isDebug and page:
        dx, dy = hp.calcAlignment(x=mouseX, y=mouseY, dw=page.getWidth(), dh=page.getHeight(), isX=True, isY=True)
        mouseDebug.keyDown("text").setFont(text="({},{})".format(round(dx, 2), round(dy, 2)))
        mouseDebug.updateAll()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            return False
        if event.type == pg.MOUSEBUTTONDOWN:
            if page != None:
                page.clicked(mouseX, mouseY)

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
