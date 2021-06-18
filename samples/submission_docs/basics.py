from manimlib import *
from manimlib.extras import *
import functools

class MoobjectCopyExample(Scene):
    def construct(self):
        triangle_1 = Triangle(fill_color=GREEN, fill_opacity=1, stroke_color=GREY)
        triangle_2 = triangle_1.copy()  # this preserves the original format of mobject that has been copied
        triangle_1.scale(3)  # formatting the original mobject like scaling later won't affect the copied bersion
        self.add(triangle_1)
        self.wait()
        circle = Circle(fill_color=PURPLE_E, fill_opacity=1, stroke_color=RED).move_to(4 * LEFT)
        self.play(ReplacementTransform(triangle_1,
                                       circle))  # transforming the original mobject won't affect the copied version
        self.play(ShowCreation(triangle_2))


class MobjectRepeatExample(Scene):
    def construct(self):
        # I have no idea what the source does, but it indeed makes the animations look cooler
        squ = Square(color=RED, fill_opacity=1)
        self.add(squ)
        self.wait()
        self.play(squ.animate.rotate(PI / 8).repeat(3))
        self.play(squ.animate.scale(2).repeat(3))
        squ_copy = squ.copy()
        self.play(squ.animate.scale(0.5).shift(2 * LEFT).repeat(5),
                  squ_copy.animate.scale(0.5).shift(2 * RIGHT).repeat(5))
        self.wait()


class MobjectMatchCoordExample(Scene):
    def construct(self):
        cir = Circle(fill_opacity=1).to_corner(UL).scale(0.5)
        squ = Square(color=BLUE, fill_opacity=1)
        self.add(squ, cir)
        self.wait()
        self.play(cir.animate.match_coord(squ, 0))
        self.play(cir.animate.match_coord(squ, 1, direction=UP))
        self.play(cir.animate.match_coord(squ, 1))
        self.wait()


class MobjectMatchColorExample(Scene):
    def construct(self):
        cir = Circle(color=BLUE, fill_opacity=1).shift(2 * LEFT)
        squ = Square(color=GREEN, fill_opacity=1).shift(2 * RIGHT)
        self.add(cir, squ)
        self.wait()
        self.play(cir.animate.match_color(squ))
        self.wait(2)


class MobjectGetEdgeExample(Scene):
    def construct(self):
        cir = Circle().shift(3 * LEFT)
        tri = Triangle()
        squ = Square().shift(3 * RIGHT)
        for shape in [cir, tri, squ]:
            self.add(
                shape,
                Dot(shape.get_bottom(), color=BLUE),
                Dot(shape.get_left(), color=RED),
                Dot(shape.get_right(), color=GREEN),
                Dot(shape.get_top(), color=WHITE),
            )


class MobjectAlignOnBorderExample(Scene):
    def construct(self):
        square_center = Square(color=RED, fill_opacity=1)
        square_inner = []
        square_outer = []
        square_inner.append(Square(side_length=0.2, fill_opacity=1, color=TEAL).align_to(square_center, RIGHT))
        square_inner.append(Square(side_length=0.2, fill_opacity=1, color=GREEN).align_to(square_center, UP))
        square_inner.append(Square(side_length=0.2, fill_opacity=1, color=YELLOW).align_to(square_center, LEFT))
        square_inner.append(Square(side_length=0.2, fill_opacity=1, color=PURPLE).align_to(square_center, DOWN))
        self.add(square_center, *square_inner)


# Move to some border
class MobjectAlignOnBorderExample2(Scene):
    def construct(self):
        cir = Circle()
        self.play(cir.animate.align_on_border(RIGHT))
        self.play(cir.animate.align_on_border(LEFT))
        self.play(cir.animate.align_on_border(UP))


class MobjectAddToBackExample(Scene):
    def construct(self):
        mobject = Mobject()
        dot_l = Circle(color=BLUE, fill_opacity=1).shift(LEFT*0.1)
        dot_r = Square(color=GREEN, fill_opacity=1).shift(RIGHT*0.1)
        mobject.add(dot_l)
        mobject.add_to_back(dot_r)
        self.play(mobject.animate.arrange())
        mobject.remove(dot_l)
        mobject.add_to_back(dot_l)
        self.play(mobject.animate.arrange())
        self.wait()


class StretchAboutPointExample(Scene):
    def construct(self):
        square = Square(stroke_color=BLUE_E, stroke_opacity=0.5, fill_color=BLUE_C, fill_opacity=1)
        # dim = 0 stretches the mobject horizontally
        # dim= 1 stretched the mobject vertically
        self.play(square.animate.stretch_about_point(2, dim=0, point=square.get_left()))
        self.play(square.animate.stretch_about_point(2, dim=1, point=ORIGIN))


class StretchInPlaceExample(Scene):
    def construct(self):
        circle = Circle(fill_color=ORANGE, fill_opacity=1)
        # dim = 0 stretches the mobject horizontally
        # dim= 1 stretched the mobject vertically
        self.play(circle.animate.stretch_in_place(3, dim=1))  # vertical as dim = 1
        self.play(circle.animate.stretch_in_place(3, dim=0))  # horizontal as dim = 0


class BraceBetweenPointsExample(Scene):
    def construct(self):
        number_plane = NumberPlane(faded_line_ratio=False)
        self.add(number_plane)
        # Brace opens upwards as points are given from left to right and vice versa, opens leftwards when points are given from down to up and vice versa
        brace_1 = BraceBetweenPoints(number_plane.coords_to_point(1, 2), number_plane.coords_to_point(4, 3),
                                     color=BLUE)  # points are given from left to right
        brace_2 = BraceBetweenPoints(number_plane.coords_to_point(-2, 1), number_plane.coords_to_point(-4, 3),
                                     color=GREY)  # points are given from right to left
        self.add(brace_1, brace_2)


class BackgroundRectangleExample(Scene):
    def construct(self):
        text = TexText("manim ", "looks ", "good")
        # text = TexText("manim ", "looks ", "good")
        text = Text("abcd")
        bg_rect = BackgroundRectangle(text[0], color=BLUE, fill_opacity=0.5, stroke_color=RED, stroke_width=1,
                                      stroke_opacity=1)
        self.add(text)
        self.play(FadeIn(bg_rect), run_time=4)


class UnderLine(Scene):
    def construct(self):
        m = TexText("M")  # Starting letter
        man = TexText("M", "anim")  # Full Word
        # m = Text("M")  # Starting letter
        # man = Text("Manim")  # Full Word
        ul = Underline(man)  # Underlining the word
        self.play(Write(m), run_time=1.5)
        self.play(ReplacementTransform(m, man[0]), FadeIn(man[1]))  # Replaces the starting letter to word
        self.play(ShowCreation(ul))  # Creates the underline
        self.wait()


class SurroundingRectangleExample(Scene):
    def construct(self):
        text = TexText("manim")
        # text = Text("manim", font="Arial")
        rect_surrounding = SurroundingRectangle(text, color=BLUE_E, buff=0.5, fill_color=BLUE_E)
        # rect_surrounding.set_fill(GREEN)
        self.add(text)
        self.play(ShowCreation(rect_surrounding), run_time=3)

        sphere = Sphere()
        mesh = SurfaceMesh(sphere)

        self.add(sphere)
        self.add(mesh)


class CurvedArrowExample(Scene):
    def construct(self):
        # Arrow tips of various shapes(filled/unfilled) can be made by importing relevant arrow types
        curved_arrow = ArcBetweenPoints(start=np.array([-2, -1, 0]), end=np.array([3, 2, 0]), color=PURPLE)
        tip_style = {'fill_opacity': 0.5, 'stroke_width': 2, 'fill_color': GREEN},
        dot = Dot(opacity=0.5, stroke_width=2, fill_color=GREEN, stroke_color=PURPLE)
        f_always(dot.move_to, lambda: curved_arrow.get_end())
        self.add(curved_arrow)
        self.add(dot)

        nmp = NumberPlane()
        self.add(nmp)


class BraceLabelExample(Scene):
    def construct(self):
        triangle = Triangle()
        bracelabel_h = BraceLabel(triangle, 'height', RIGHT)
        bracelabel_w = BraceLabel(triangle, 'base')

        self.add(triangle)
        self.add(bracelabel_h)
        self.add(bracelabel_w)


class ArrowTrace1(Scene):
    CONFIG = {
        "camera_config": {"background_color": WHITE}
    }
    def construct(self):
        nmp = NumberPlane(faded_line_ratio=0)
        self.add(nmp)
        path = ParametricCurve(lambda x: [x, 2+np.sin(x), 0], t_range=[-PI, PI, 0.9], use_smoothing=False)
        self.add(path)
        vect = Vector()
        dot = Dot().scale(0)
        vect.add_updater(lambda v: v.put_start_and_end_on(ORIGIN, dot.get_center()))
        self.add(vect)
        self.play(MoveAlongPath(dot, path, t_range=[0, PI, 0.9]), run_time=5)
        self.wait()
        dummy = 1
