import pygame as pg
# from pygame import gfxdraw as pgx
import helper as hp

FPS = 50
width = 1050
height = 700
title = "Teaching APP"
isDebug = False

# Cannot import from graphics
backgroundColor = ((50, 50, 50))
dragObj = None


def update():
    global mouseDebug, dragObj, g
    # print("START")
    # background = pg.Surface(self.size)
    # background.fill(hp.red)

    # INPUT======================================
    mouseX, mouseY = pg.mouse.get_pos()
    if isDebug and page:
        dx, dy = hp.calcAlignment(x=mouseX, y=mouseY, dw=page.getWidth(), dh=page.getHeight(), isX=True, isY=True)
        mouseDebug.keyDown("text").setFont(text="({},{})".format(round(dx, 2), round(dy, 2)))
        mouseDebug.updateAll()
    if dragObj != None:
        dx, dy = hp.calcAlignment(x=mouseX - dragObj.container.x - dragObj.getWidth() // 2, y=mouseY - dragObj.container.y - dragObj.getHeight() // 2, dw=dragObj.container.getWidth() -
                                  dragObj.getWidth(), dh=dragObj.container.getHeight() - dragObj.getHeight(), isX=True, isY=True)
        dragObj.setAlignment(dx=dx, dy=dy)
        dragObj.updateAll()

    if page != None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            elif event.type == pg.VIDEORESIZE:
                page.setSize(width=event.w, height=event.h)
                page.updateAll()
            #     g = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
            elif event.type == pg.MOUSEBUTTONDOWN:
                view = page.clicked(mouseX, mouseY)
                if view != None and view != False and view.isDraggable:
                    dragObj = view

                if event.button == 4:
                    page.scrollUp()
                elif event.button == 5:
                    page.scrollDown()

            elif event.type == pg.MOUSEBUTTONUP:
                if dragObj != None:
                    container = page.getEmptyContainer(mouseX, mouseY)
                    if container != None and page.canDragView(dragObj, container):
                        dragObj.setContainer(container)
                    dragObj.setAlignment(dx=0.0, dy=0.0)
                    dragObj.container.updateAll()
                    dragObj = None

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
g = pg.display.set_mode(size, pg.RESIZABLE)
setPage(None)


if __name__ == '__main__':
    hp.clear()
    print("RUNNING GUI Main")


"""

"""
