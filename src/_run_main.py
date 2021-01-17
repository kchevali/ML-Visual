from gui import createGUI, runGUI
import helper as hp


if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    createGUI()

    from graphics import ZStack
    from page import CompPage
    runGUI(ZStack(items=CompPage()))
