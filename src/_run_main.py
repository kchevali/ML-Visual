
import gui
import helper as hp
from page import *
import traceback
from time import time

if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    try:
        startTime = time()
        page = MainPage()
        midTime = time()
        gui.setPage(page)
        print("Page Gen Time: {}s".format(round(midTime - startTime, 2)))
        print("Loading Time: {}s".format(round(time() - midTime, 2)))
        while gui.update():
            pass

    except:
        print("ERROR - program terminated")
        print(traceback.format_exc())


"""

"""
