
from table import Table
from label import Label
from gui import GUI
import helper as hp
from picker import Picker


if __name__ == '__main__':
    hp.clear()
    print("Running MAIN")

    from gui import GUI
    gui = GUI("Table MAIN", width=1000, height=650)

    modelMenu = gui.newWindow()
    print("DX:", modelMenu.dx, "DY:", modelMenu.dy)
    modelMenu.splitFromFile("pages/modelPage.json")
    modelMenu["NW"].setObject(Label("Decision Tree"))
    modelMenu["NE"].setObject(Label("KNN"))
    modelMenu["SW"].setObject(Label("Model 3"))
    modelMenu["SE"].setObject(Label("Model 4"))
    # modelMenu.splitWindow2D(2)

    runPage = gui.newWindow()
    runPage.splitFromFile("pages/runPage.json")

    # Header
    # runPage.splitWindow(2, isVertical=True, ratios=[0.1, 0.9], names=["header", ])

    # # File Explorers
    # runPage[1].splitWindow(2, isVertical=False, ratios=[0.15, 0.85])

    # # Hyper Parameters
    # runPage[1][1].splitWindow(2, isVertical=True, ratios=[0.6, 0.4])

    # # Table/Data
    # runPage[1][1][0].splitWindow(2, isVertical=False, ratios=[0.7, 0.3])
    # runPage["data"].setObject(Table.readCSV("examples/decisionTree.csv"))
    runPage["header"].setObject(Label("Run Page"), dx=-1.0)
    runPage["file"].setObject(Picker(isVertical=True, entries=hp.getFiles("examples"), entryHeight=40), dy=-1.0)

    gui.setWindow(runPage)

    # ratio = 1.0
    # dr = -0.01
    while gui.update():
        pass
        # gui.window.setRatios(ratios=[ratio, 1.0 - ratio])
        # gui.window[1].setRatios(ratios=[ratio, 1.0 - ratio])
        # ratio += dr
        # if ratio < 0.0 or ratio > 1.0:
        #     dr *= -1
        #     ratio = 0.0 if ratio < 0.0 else 1.0
    gui.close()


"""

"""
