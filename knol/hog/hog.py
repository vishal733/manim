from manimlib import *


class Hog1(Scene):
    def construct(self):
        txt = TexText("""
            HOG - Histogram of Oriented Gradients
        """, math_mode=False).scale(1.2).shift(UP*3.4)
        txt.set_stroke(BLACK, 1, background=True)
        ln = Line(LEFT * 5.5, RIGHT * 5.5).shift(UP*3)
        ln.set_fill(BLACK, 1)

        self.add(txt)
        self.play(FadeIn(txt), FadeIn(ln), run_time=1)
        self.wait(1)

        txt2 = Text("- Provides features for object recognition").scale(0.7).move_to(2*UP+6*LEFT, aligned_edge=LEFT_SIDE)
        self.add(txt2)

        txt3 = Text("- Simple and effective").scale(0.7).move_to(UP + 6 * LEFT, aligned_edge=LEFT_SIDE)
        self.add(txt3)

        txt4 = Text("- Authors: Dalal and Triggs, 2005").scale(0.7).move_to(6 * LEFT, aligned_edge=LEFT_SIDE)
        self.add(txt4)

        return


class Hog2(Scene):
    def construct(self):
        txt = TexText("""
                    HOG - Histogram of Oriented Gradients
                """, math_mode=False, font=default_font).scale(1.2).shift(UP * 3.4)
        txt.set_stroke(BLACK, 1, background=True)
        ln = Line(LEFT * 5.5, RIGHT * 5.5).shift(UP * 3)
        ln.set_fill(BLACK, 1)
        self.add(txt, ln)


default_font = 'Apercu Mono Pro'  # You probably want to change this to a free font
class MyObj(VMobject):
    @property
    def digit_str(self):
        return str(self.value)

    @property
    def digit_str_sq(self):
        return str(self.value**2)

    def __init__(self, value, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.value = value
        txt = Text(self.digit_str, font=default_font)
        txt2 = Text(self.digit_str_sq).scale(0.5).shift(DOWN/2)
        self.add(txt)
        self.add(txt2)
        txt2 = Text(self.digit_str_sq, font=default_font).scale(0.5)
        self.add(txt2.shift(DOWN/2))
        self.add(txt2.shift(DOWN/2).copy().set_fill(BLUE))
        self.add(txt2.shift(DOWN/2).copy().set_fill(GREEN))
        self.add(txt2.shift(DOWN / 2).copy().set_fill(YELLOW))
        self.add(txt2.shift(DOWN / 2).copy().set_fill(RED))
        self.add(txt2.shift(DOWN / 2).copy().set_fill(PINK))
        return


class Practise(Scene):
    def construct(self):
        objs = VGroup()
        for i in range(10):
            obj = MyObj(i)
            objs.add(obj)
        objs.arrange(RIGHT)
        self.add(objs)
        objs.move_to(LEFT*5, aligned_edge=LEFT_SIDE)
        # print("Centered height: {}".format(objs[0].centered_height))
        return
