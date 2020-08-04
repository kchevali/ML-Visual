from frame import Frame
import helper as hp


class Label(Frame):
    def __init__(self, text):
        super().__init__(isWidthFixed=False, isHeightFixed=False)
        self.text = text

    def display(self):
        if self.g == None:
            return
        super().display()

        surface, textWidth, textHeight = hp.getFont(self.text, self.width, self.height, hp.red)
        pos = hp.findPosition(textWidth, textHeight, self.x, self.y, self.width, self.height, self.parent.dx, self.parent.dy)
        self.g.blit(surface, pos)
