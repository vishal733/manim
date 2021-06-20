import os, sys
from manimlib import *
from lib.ScreenGrid import ScreenGrid


class Scene(Scene):
    CONFIG = {
        "camera_config": {"background_color": "#161616"},
        "include_grid": True
    }

    def setup(self):
        if self.include_grid:
            self.add(ScreenGrid().fade(0.7))


class AddUpdater1(Scene):
    def construct(self):
        dot = Dot()
        text = Text("Label")
        text.next_to(dot, RIGHT, buff=MED_SMALL_BUFF)
        self.add(dot, text)

        def update_text(obj):
            obj.next_to(dot, RIGHT, buff=MED_SMALL_BUFF)

        text.add_updater(update_text)
        self.play(dot.shift, UP * 2)
        self.wait(1)
        self.play(dot.shift, LEFT * 3)
        text.remove_updater(update_text)
        self.wait()


# Update function as a lambda function
class AddUpdater2(Scene):
    def construct(self):
        dot = Dot()
        text = Text("Label") \
            .next_to(dot, RIGHT, buff=MED_SMALL_BUFF)
        self.add(dot, text)
        text.add_updater(lambda m: m.next_to(dot, RIGHT, buff=SMALL_BUFF))
        self.play(dot.shift, UP * 2)
        self.wait(1)
        self.play(dot.shift, LEFT * 3)
        text.clear_updaters()  # NOTE: Take note of this function
        self.wait()


# updater function only applied within a single "play" scope
class AddUpdater3(Scene):
    def construct(self):
        dot = Dot()
        text = Text("Label") \
            .next_to(dot, RIGHT, buff=MED_SMALL_BUFF)
        self.add(dot, text)

        def update_text(obj):
            obj.next_to(dot, RIGHT, buff=MED_SMALL_BUFF)

        # text.add_updater(lambda m: m.next_to(dot, RIGHT, buff=SMALL_BUFF))
        self.play(ApplyMethod(dot.shift, UP * 2), UpdateFromFunc(text, update_text))
        self.play(dot.animate.shift(LEFT * 2))
        self.wait()


# Update number based on position in screen
class UpdateNumber(Scene):
    def construct(self):
        number_line = NumberLine(x_range=[-1, 1])
        triangle = RegularPolygon(3, start_angle=-PI / 2).scale(0.2) \
            .move_to(number_line.get_left()).shift(0.2 * UP)
        decimal = DecimalNumber(
            0,
            num_decimal_places=2,
            include_sign=True,
            unit="\\rm cm"
        ).scale(0.5)
        decimal.add_updater(lambda d: d.next_to(triangle, UP * 1.0))
        decimal.add_updater(lambda d: d.set_value(triangle.get_center()[0]))
        self.add(number_line, triangle, decimal)

        self.play(
            triangle.animate.shift(number_line.get_right() * 2),
            rate_func=there_and_back,
            run_time=5
        )
        return


# Reverse: Using value, update something else on the screen
class UpdateValueTracker1(Scene):
    def construct(self):
        theta = ValueTracker(PI / 2)
        line_1 = Line(ORIGIN, RIGHT * 3, color=RED)
        line_2 = Line(ORIGIN, RIGHT * 3, color=GREEN)

        line_2.rotate(theta.get_value(), about_point=ORIGIN)
        line_2.add_updater(
            lambda m: m.set_angle(theta.get_value())
        )
        self.add(line_1, line_2)
        self.play(theta.animate.increment_value(PI / 2))
        self.wait()
        return


# become method
class UpdateValueTracker2(Scene):
    CONFIG = {
        "line_1_color": ORANGE,
        "line_2_color": PINK,
        "lines_size": 3.5,
        "theta": PI / 2,
        "increment_theta": PI / 2,
        "final_theta": PI,
        "radius": 0.7,
        "radius_color": YELLOW
    }

    def construct(self):
        theta = ValueTracker(self.theta)
        line_1 = Line(ORIGIN, RIGHT * self.lines_size, color=self.line_1_color)
        line_2 = Line(ORIGIN, RIGHT * self.lines_size, color=self.line_2_color)

        line_2.rotate(theta.get_value(), about_point=ORIGIN)
        line_2.add_updater(
            lambda m: m.set_angle(theta.get_value())
        )
        angle = Arc(
            radius=self.radius,
            start_angle=line_1.get_angle(),
            angle=line_2.get_angle(),
            color=self.radius_color
        )

        self.play(*[ShowCreation(obj) for obj in [line_1, line_2, angle]])

        angle.add_updater(lambda d: d.become(Arc(
            radius=self.radius, start_angle=line_1.get_angle(),
            angle=line_2.get_angle(), color=self.radius_color
        )))

        self.play(theta.animate.increment_value(PI / 2), rate_func=there_and_back)


# Order of animation matters
class ToEdgeAnimation1(Scene):
    def construct(self):
        mob = Circle()
        self.add(mob)

        self.play(mob.animate.scale(0.1).to_edge(UP, 0))
        self.wait()


# Generate target even before adding it
# MoveToTarget method
class ToEdgeAnimation2(Scene):
    def construct(self):
        mob = Circle()
        mob.generate_target()
        mob.target.scale(0.1)
        mob.target.to_edge(RIGHT, buff=0)

        self.add(mob)
        self.play(MoveToTarget(mob))
        self.wait()


class ScaleAnimation(Scene):
    def construct(self):
        mob = Circle()
        dot = Dot([6, 0, 0])

        self.add(mob, dot)
        self.play(mob.scale, 3)

        self.play(
            mob.scale, 1 / 3, {"about_point": dot.get_center()}
        )
        self.wait()


class ArrangeAnimation1(Scene):
    def construct(self):
        vgroup = VGroup(
            Square(), Circle()
        )
        self.add(vgroup)
        self.wait()
        self.play(vgroup.arrange, DOWN, {"buff": 0})
        self.wait()


class ArrangeAnimation3(Scene):
    def construct(self):
        vgroup = VGroup(
            Square(), Circle()
        )
        text = Text("Hello World").to_corner(UL)
        self.add(vgroup)
        self.wait()
        self.play(vgroup.animate.arrange(DOWN, {"buff": 0}), Write(text))
        self.wait()


class ShiftAnimation1(Scene):
    def construct(self):
        mob = Circle()

        def modify(obj):
            obj.shift(LEFT)
            obj.set_color(TEAL)
            return obj

        self.add(mob)
        # self.play(mob.animate.shift(LEFT).set_color(TEAL))
        self.play(ApplyFunction(modify, mob))
        self.wait()


# To modify both contents inside VGroup and an element inside it, use ApplyFunction
class MultipleAnimationVGroup(Scene):
    def construct(self):
        rect, circ = Rectangle(), Circle()
        vgroup = VGroup(rect, circ)

        def modify(vg):
            r, c = vg
            r.set_height(1)
            vg.arrange(DOWN, buff=0)
            return vg

        self.add(vgroup)
        self.play(ApplyFunction(modify, vgroup))
        # self.play(rect.animate.set_height(1), vgroup.animate.arrange(DOWN, buff=0))
        self.wait()


# Big limitation of methods is that they are linear (they perform a single transformation from start to finish without intermediate points).
# Comparing classical rotation with rotation using methods
# The intermediate points are not seen using methods (white square - square1)
class RotationAnimationFail(Scene):
    def construct(self):
        square1, square2 = VGroup(Square(), Square(color=TEAL)).scale(0.3).set_y(-3)
        reference = DashedVMobject(Circle(radius=3, color=GREY), num_dashes=100)
        self.add(square1, square2, reference)

        self.play(
            square1.animate.rotate(2 * PI / 3, about_point=ORIGIN),
            Rotate(square2, 2 * PI / 3, about_point=ORIGIN),
            run_time=4
        )
        # One Fix below
        self.play(
            Rotate(square1, 2 * PI / 3, about_point=ORIGIN),
            Rotate(square2, 2 * PI / 3, about_point=ORIGIN),
            run_time=4
        )
        self.wait()


class RotationAndMoveFail(Scene):
    def construct(self):
        square1, square2 = VGroup(Square(color=RED), Square(color=BLUE)).scale(0.5).set_x(-5)
        reference = DashedVMobject(Line(LEFT * 5, RIGHT * 5, color=GREY))
        self.add(square1, square2, reference)

        square2.save_state()

        def update_rotate_move(mob, alpha):
            print("alpha: {}".format(alpha))
            square2.restore()
            square2.shift(RIGHT * 10 * alpha)
            square2.rotate(3 * PI * alpha)

        self.play(
            square1.animate.rotate(3 * PI).move_to(np.array([5, 0, 0])),
            UpdateFromAlphaFunc(square2, update_rotate_move),
            run_time=4
        )


# Creating a custom animation function:
# Setting about_edge to None creates issue in this code
class ShiftAndRotate(Animation):
    CONFIG = {
        "axis": OUT,
        "run_time": 5,
        "rate_func": smooth,
        "about_point": None,
        "about_edge": None
    }

    def __init__(self, mobject, direction, radians, **kwargs):
        assert (isinstance(mobject, Mobject))
        digest_config(self, kwargs)
        self.mobject = mobject
        self.direction = direction
        self.radians = radians

    def interpolate_mobject(self, alpha):
        print("self.about_edge: {}".format(self.about_edge))
        self.mobject.become(self.starting_mobject)
        self.mobject.shift(alpha * self.direction)
        self.mobject.rotate(
            alpha * self.radians,
            axis=self.axis,
            about_point=self.about_point,
            # about_edge=self.about_edge
        )


class RotationAndMove(Scene):
    def construct(self):
        square1, square2 = VGroup(
            Square(color=RED), Square(color=BLUE)
        ).scale(0.5).set_x(-5)

        reference = DashedVMobject(Line(LEFT * 5, RIGHT * 5, color=GREY))
        self.add(square1, square2, reference)
        self.play(
            square1.animate.rotate(3 * PI).move_to(np.array([5, 0, 0])),
            ShiftAndRotate(square2, RIGHT * 10, 3 * PI),
            run_time=4
        )
        self.wait()


# Working with arbitrary path
# path.point_from_proportion()
class RotateWithPath(Scene):
    def construct(self):
        square1, square2 = VGroup(
            Square(color=RED), Square(color=BLUE)
        ).scale(0.5).set_x(-5)

        path = ArcBetweenPoints(LEFT*5, RIGHT*5, angle=-PI/2, stroke_opacity=0.5)
        # path.points[1:3] += UP*2

        square2.save_state()
        def update_rotate_move(mob, alpha):
            square2.restore()
            square2.move_to(path.point_from_proportion(alpha))
            square2.rotate(3*PI*alpha)

        self.add(square1, square2, path)
        self.play(
            MoveAlongPath(square1, path),
            Rotate(square1, 2*PI/3, about_point=square1.get_center()),
            UpdateFromAlphaFunc(square2, update_rotate_move),
            run_time=4
        )


# You can set size of object by doing set_height
class MoveAlongPathWithAngle(Scene):
    def get_pending(self, path, proportion, dx=0.01):
        if proportion < 1:
            coord_i = path.point_from_proportion(proportion-dx)
            coord_f = path.point_from_proportion(proportion+dx)
        else:
            coord_i = path.point_from_proportion(proportion - dx)
            coord_f = path.point_from_proportion(proportion)
        line = Line(coord_i, coord_f)
        angle = line.get_angle()

        return angle

    def construct(self):
        path = ArcBetweenPoints(LEFT*5, RIGHT*5, angle=-PI/2, stroke_opacity=0.5)
        path = Square( stroke_opacity=0.5)

        start_angle = self.get_pending(path, 0)
        triangle = Triangle().set_height(0.5)
        triangle.move_to(path.get_start())
        triangle.rotate(-PI/2)

        triangle.save_state()
        triangle.rotate(start_angle, about_point=triangle.get_center())

        def update_rotate_move(mob, alpha):
            triangle.restore()
            angle = self.get_pending(path, alpha)
            triangle.move_to(path.point_from_proportion(alpha))
            triangle.rotate(angle, about_point=triangle.get_center())

        self.add(triangle, path)
        self.play(UpdateFromAlphaFunc(triangle, update_rotate_move), run_time=4)
        self.wait(3)

