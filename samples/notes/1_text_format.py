from manimlib import *
import numpy as np


# TexMobject -- Tex
# TextMobject -- TexText
class Text1(Scene):
    def construct(self):
        txt = TexText("This is a TexText")
        txt.shift(3 * UP)
        self.add(txt)

        text = Text("This is a Text")
        text.shift(2 * UP)
        self.play(Write(text))
        self.wait(1)

        formula = Tex("This is a Tex (formula)")
        formula.shift(UP)
        self.play(Write(formula))
        self.wait(1)


class Formula1(Scene):
    def construct(self):
        tipesOfText = TexText("""
                    This is a regular text,
                    $this is a formula$,
                    $$this is a formula$$
                    """)
        tipesOfText.shift(3 * UP)
        self.play(Write(tipesOfText))
        self.wait(1)

        tipesOfText = TexText("""
                    Inline and newline formula,
                    $\\frac{x}{y}$,
                    $$x^2+y^2=a^2$$
                    """)
        tipesOfText.shift(UP)
        self.play(Write(tipesOfText))
        self.wait(3)

        tipesOfText = TexText("""
                    This is a regular text,
                    $\\displaystyle\\frac{x}{y}$,
                    $$x^2+y^2=a^2$$
                    """)
        tipesOfText.shift(DOWN)
        self.play(Write(tipesOfText))


class AbsolutePosition(Scene):
    def construct(self):
        text = TexText("Up")
        text.to_edge(UP)
        self.play(Write(text))
        # self.wait(1)

        text = TexText("Down")
        text.to_edge(DOWN)
        self.play(Write(text))
        # self.wait(1)

        text = TexText("Right")
        text.to_edge(RIGHT)
        self.play(Write(text))
        # self.wait(1)

        text = TexText("""ALeft""")
        text.to_edge(LEFT)
        self.play(Write(text))
        # self.wait(1)

        text = TexText("Text")
        text.to_edge(UP + RIGHT)
        self.play(Write(text))
        # self.wait(1)

        text = TexText("LessBorderText")
        text.to_edge(DOWN + RIGHT, buff=0.1)
        self.play(Write(text))
        # self.wait(1)

        textM = TexText("Text")
        textC = TexText("Central text")
        textM.move_to(0.5 * UP)
        self.play(Write(textM), Write(textC))
        self.wait(1)
        textM.move_to(1 * UP + 1 * RIGHT)
        self.play(Write(textM))
        self.wait(1)
        textM.move_to(3 * UP + 1 * RIGHT)


class RelativePosition(Scene):
    def construct(self):
        textM = TexText("Text")
        textC = TexText("Reference text")
        textM.next_to(textC, LEFT, buff=1)
        self.play(Write(textM), Write(textC))
        self.wait(3)


class RelativePosition2(Scene):
    def construct(self):
        textM = TexText("Text")
        textC = TexText("Reference text")
        textM.shift(UP * 0.1)
        self.play(Write(textM), Write(textC))
        self.wait(3)


class RotateObject(Scene):
    def construct(self):
        textM = TexText("Text")
        textC = TexText("Reference text")
        textM.shift(UP)
        textM.rotate(PI / 4)
        self.play(Write(textM), Write(textC))
        self.wait(2)
        textM.rotate(PI / 4)
        self.wait(2)
        textM.rotate(PI / 4)
        self.wait(2)
        textM.rotate(PI / 4)
        self.wait(2)
        textM.rotate(PI)
        self.wait(2)


class FlipObject(Scene):
    def construct(self):
        textM = TexText("Text")
        textM.flip(UP)
        self.play(Write(textM))
        self.wait(2)


class SizeTextOnLaTeX(Scene):
    def construct(self):
        textHuge = TexText("{\\Huge Huge Text 012.\\#!?} Text")
        texthuge = TexText("{\\huge huge Text 012.\\#!?} Text")
        textLARGE = TexText("{\\LARGE LARGE Text 012.\\#!?} Text")
        textLarge = TexText("{\\Large Large Text 012.\\#!?} Text")
        textlarge = TexText("{\\large large Text 012.\\#!?} Text")
        textNormal = TexText("{\\normalsize normal Text 012.\\#!?} Text")
        textsmall = TexText("{\\small small Text 012.\\#!?} Texto normal")
        textfootnotesize = TexText("{\\footnotesize footnotesize Text 012.\\#!?} Text")
        textscriptsize = TexText("{\\scriptsize scriptsize Text 012.\\#!?} Text")
        texttiny = TexText("{\\tiny tiny Texto 012.\\#!?} Text normal")
        textHuge.to_edge(UP)
        texthuge.next_to(textHuge, DOWN, buff=0.1)
        textLARGE.next_to(texthuge, DOWN, buff=0.1)
        textLarge.next_to(textLARGE, DOWN, buff=0.1)
        textlarge.next_to(textLarge, DOWN, buff=0.1)
        textNormal.next_to(textlarge, DOWN, buff=0.1)
        textsmall.next_to(textNormal, DOWN, buff=0.1)
        textfootnotesize.next_to(textsmall, DOWN, buff=0.1)
        textscriptsize.next_to(textfootnotesize, DOWN, buff=0.1)
        texttiny.next_to(textscriptsize, DOWN, buff=0.1)
        self.add(textHuge, texthuge, textLARGE, textLarge, textlarge, textNormal, textsmall, textfootnotesize,
                 textscriptsize, texttiny)
        self.wait(3)


class TextFonts(Scene):
    def construct(self):
        textNormal = TexText("{Roman serif text 012.\\#!?} Text")
        textItalic = TexText("\\textit{Italic text 012.\\#!?} Text")
        textTypewriter = TexText("\\texttt{Typewritter text 012.\\#!?} Text")
        textBold = TexText("\\textbf{Bold text 012.\\#!?} Text")
        textSL = TexText("\\textsl{Slanted text 012.\\#!?} Text")
        textSC = TexText("\\textsc{Small caps text 012.\\#!?} Text")
        textNormal.to_edge(UP)
        textItalic.next_to(textNormal, DOWN, buff=.5)
        textTypewriter.next_to(textItalic, DOWN, buff=.5)
        textBold.next_to(textTypewriter, DOWN, buff=.5)
        textSL.next_to(textBold, DOWN, buff=.5)
        textSC.next_to(textSL, DOWN, buff=.5)
        self.add(textNormal, textItalic, textTypewriter, textBold, textSL, textSC)
        self.wait(3)
