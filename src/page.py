from graphics import *
import helper as hp


class DefaultPage(Container):
    def __init__(self):
        view = Color(hp.backgroundColor)
        super().__init__(view=view)


class RunPage(Container):
    def __init__(self):
        view = VStack(views=[
            Label("Title"),
            HStack(views=[
                VStack(views=[
                    Label("Example 1"),
                    Label("Example 2")
                ]),
                VStack(views=[
                    HStack(views=[
                        Color(hp.red),
                        Color(hp.green)
                    ]),
                    Color(hp.backgroundColor)
                ])
            ])
        ])
        super().__init__(view=view)
