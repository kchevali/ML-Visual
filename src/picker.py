from frame import Frame
import helper as hp
from table import Table


class Picker(Frame):

    def __init__(self, isVertical, entries=None, entryHeight=100):
        super().__init__(isWidthFixed=False, isHeightFixed=True)
        self.entries = []
        self.index = None
        self.isVertical = isVertical
        self.entryHeight = entryHeight

        if entries != None:
            self.addAll(entries)

    def add(self, entry):
        if len(self) == 0:
            self.index = 0
        self.entries.append(entry)
        self.height = len(self) * self.entryHeight

    def addAll(self, entries):
        for entry in entries:
            self.add(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        return self.entries[key]

    def clicked(self, x, y):
        if self.g == None or len(self) == 0:
            return
        length, click = (self.height, y) if self.isVertical else (self.width, x)
        clickedIndex = int((click - self.y) * len(self) // length)
        if self.parent != None and clickedIndex >= 0 and clickedIndex < len(self):
            self.parent.root["data"].setObject(Table.readCSV("examples/{}".format(self[clickedIndex])))
        # print("Clicked:", clickedIndex)

    def display(self):
        if self.g == None or len(self) == 0:
            return
        super().display()
        entryWidth, entryHeight = (self.width, self.height / len(self)) if self.isVertical else (self.width / len(self), self.height)
        isX, isY = (0, 1) if self.isVertical else (1, 0)
        x, y = (self.x, self.y)
        for entry in self.entries:
            text, textWidth, textHeight = hp.getFont(entry, entryWidth, entryHeight, hp.red)
            pos = hp.findPosition(textWidth, textHeight, x, y, entryWidth, entryHeight)
            self.g.blit(text, pos)
            x += entryWidth * isX
            y += entryHeight * isY
