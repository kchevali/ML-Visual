import traceback
from time import time
from gui import createGUI, runGUI
import helper as hp
import pygame as pg

if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    try:
        createGUI()
        from page import *
        runGUI(ZStack(items=MenuPage()))

    except:
        print("ERROR - program terminated")
        print(traceback.format_exc())


"""

"""
