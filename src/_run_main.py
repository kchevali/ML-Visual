from gui import createGUI, runGUI
import helper as hp
import os


if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    createGUI()

    from graphics import ZStack
    from page import MenuPage
    runGUI(ZStack(items=MenuPage()))
