import pygame as pg
import pygame.freetype
# from pygame import gfxdraw as pgx
import helper as hp

if not 'g' in globals():
    g = None
    width, height = 0, 0


def createGUI():
    global width, height, g
    width = 1050
    height = 700
    pg.init()
    # pg.font.init()
    pg.freetype.init()
    g = pg.display.set_mode((width, height), pg.RESIZABLE)


def runGUI(page):
    print("Loading...", end="\r")
    title = "Teaching APP"
    pg.display.set_caption(title)
    clock = pg.time.Clock()

    FPS = 50
    isDebug = False
    mouseDebug = None
    dragObj = None

    # Cannot import from graphics
    backgroundColor = ((50, 50, 50))

    if isDebug:
        from page import createMouseDebug
        mouseDebug = createMouseDebug()
        page.addView(mouseDebug)
    page.setSize(width=width, height=height)
    page.updateAll()
    print("Loading Complete!")

    # print("START")
    # background = pg.Surface(size)
    # background.fill(hp.red)

    # INPUT======================================
    isRun = True
    while isRun:
        mouseX, mouseY = pg.mouse.get_pos()
        if isDebug:
            dx, dy = hp.calcAlignment(x=mouseX, y=mouseY, dw=width, dh=height, isX=True, isY=True)
            mouseDebug.keyDown("text").setFont(text="({},{})".format(round(dx, 2), round(dy, 2)))
            mouseDebug.updateAll()

        page.update()
        # moves drag obj to mouse
        if dragObj != None:
            dx, dy = hp.calcAlignment(x=mouseX - dragObj.container.x - dragObj.getWidth() // 2, y=mouseY - dragObj.container.y - dragObj.getHeight() // 2, dw=dragObj.container.getWidth() -
                                      dragObj.getWidth(), dh=dragObj.container.getHeight() - dragObj.getHeight(), isX=True, isY=True)
            dragObj.setAlignment(dx=dx, dy=dy)
            dragObj.updateAll()

        page.hoverMouse(mouseX, mouseY)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                isRun = False

            if event.type == pg.VIDEORESIZE:
                page.setSize(width=event.w, height=event.h)
                page.updateAll()
                g = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    view = page.clicked(mouseX, mouseY)
                    if view != None and view != False and view.isDraggable:
                        dragObj = view
                elif event.button == 4:
                    page.scrollUp()
                elif event.button == 5:
                    page.scrollDown()

            elif event.type == pg.MOUSEBUTTONUP:
                if dragObj != None:
                    for container in page.getEmptyContainers(mouseX, mouseY):
                        if container != None and page.canDragView(dragObj, container):
                            dragObj.setContainer(container)
                            page.draggedView(dragObj)
                            break
                    dragObj.setAlignment(dx=0.0, dy=0.0)
                    dragObj.container.updateAll()
                    dragObj = None

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
