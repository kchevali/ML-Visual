
import gui
import helper as hp
from page import *
import traceback

if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    try:
        page = RunPage()
        gui.setPage(page)
        # c = page.view.containers
        # a = c[0].view
        # dx = 0.01
        # dy = 0.01
        while gui.update():
            pass
            # if abs(a.dx + dx) >= 1:
            #     dx *= -1
            # if abs(a.dy + dy) >= 1:
            #     dy *= -1
            # for con in c:
            #     con.view.setAlignment(dx=con.view.dx + dx, dy=con.view.dy + dy)

    except:
        print("ERROR - program terminated")
        print(traceback.format_exc())


"""

"""
