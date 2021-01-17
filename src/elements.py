from graphics import ZStack, Rect, Label, Button, Color


def createLabel(text, color=Color.steelBlue, views=[], **kwargs):
    return ZStack([
        Rect(color=color, cornerRadius=10),
        Label(text)
    ] + views, **kwargs)


def createButton(text, color=Color.steelBlue, views=[], **kwargs):
    return Button([
        Rect(color=color, cornerRadius=10),
        Label(text)
    ] + views, **kwargs)
