import traceback
from time import time
importStart = time()
import gui
guiTime = time()
import helper as hp
helperTime = time()
from page import *
pageTime = time()

if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    try:
        startTime = time()
        page = MainPage()
        midTime = time()
        gui.setPage(page)
        print("Import Time: {}".format(round(pageTime - importStart, 2)))
        print("Page Gen Time: {}s".format(round(midTime - startTime, 2)))
        print("Loading Time: {}s".format(round(time() - midTime, 2)))
        while gui.update():
            pass

    except:
        print("ERROR - program terminated")
        print(traceback.format_exc())


"""

"""
