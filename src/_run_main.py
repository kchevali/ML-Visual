from gui import createGUI, runGUI
import helper as hp
import os


if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    curr_path = os.path.abspath(os.curdir)
    if(curr_path.endswith("src")):
        print("Warning: Script is running from src - moving back a directory")
        os.chdir("..")

    createGUI()

    from graphics import ZStack
    from page import MenuPage
    runGUI(ZStack(items=MenuPage()))
