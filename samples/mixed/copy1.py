from manimlib import *

# Font to be used - "LM Roman 10"
FONT = "LM Roman 10"


class MultipleSqures(VGroup):
    def __init__(self, count, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.squares = VGroup(*[Square().set_stroke(RED).flip() for i in range(count)])
        self.squares.arrange(RIGHT, buff=0)
        # sq = Square().set_stroke(RED)
        self.add(self.squares)

    def addText(self, l_txts):
        return


class Copy1(Scene):
    def construct(self):
        # heading = TexText("Electric Flux").scale(2).to_corner(TOP, buff=0.1)
        # self.play(Write(heading))
        #
        # texx = Tex(r"\Phi=\oint\vec{E}\cdot d\vec{A}=\frac{Q}{\varepsilon_{o}}")
        # self.play(Write(texx))
        #
        # self.clear()

        # heading = TexText(r"Arrays").scale(2).set_fill(RED)
        # self.play(Write(heading), play_time=3)
        # self.play(heading.animate.to_corner(TOP, buff=0.1))

        text1 = Text("thing[ ]", font=FONT, t2c={"thing": PINK, "[": RED, "]": RED})
        self.play(Write(text1), play_time=1)
        # self.wait()
        self.play(text1.animate.shift(4*LEFT), play_time=1)

        text2 = Text("=", font=FONT).next_to(text1, RIGHT)
        self.play(Write(text2), play_time=1)

        squares = MultipleSqures(4).shift(RIGHT*2.3)
        self.play(Write(squares))

        l_txts = ["Thing1", "Thing2", "Thing3", "Thing4"]
        squares.addText(l_txts)

        squares.squares[0].round_corners(0.1)

        return
