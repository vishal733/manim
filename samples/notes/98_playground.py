from manimlib import *
import math
import random


class Vishal(Scene):
    def construct(self):
        obj = Axes()
        obj.add_coordinate_labels()
        self.add(obj)
        self.wait(2)
        self.remove(obj)

        obj = ThreeDAxes()
        obj.add_coordinate_labels()
        self.add(obj)
        self.wait(2)
        self.remove(obj)

        obj = NumberPlane()
        obj.add_coordinate_labels()
        self.add(obj)
        self.wait(2)
        self.remove(obj)

        obj = ComplexPlane()
        obj.add_coordinate_labels()
        self.add(obj)
        self.wait(2)
        # self.remove(obj)
        return


class Vishal3(Scene):
    def construct(self):
        rect = Rectangle(width=9, height=0.5).shift(DOWN*2).set_color(WHITE).set_opacity(1.0)
        self.add(rect)

        rect_vert = Rectangle(width=0.5, height=4).shift(LEFT*3).set_color(WHITE).set_opacity(1.0)
        self.add(rect_vert)

        rect_vert = rect_vert.copy().move_to(ORIGIN)
        self.add(rect_vert)

        rect_vert = rect_vert.copy().move_to(ORIGIN+3*RIGHT)
        self.add(rect_vert)

        # torus = Torus(r1=1, r2=0.5).rotate(PI/2, np.array((1., 0., 0.)))
        # self.add(torus)

        # cylinder = Cylinder()
        # self.add(cylinder)
        #
        # sq3d = Square3D()
        # self.add(sq3d)

        cube = Cube()
        self.add(cube)

        light = self.camera.light_source
        self.add(light)
        light.save_state()
        # self.play(light.animate.move_to(3 * IN), run_time=5)
        # self.play(light.animate.shift(10 * OUT), run_time=5)

        self.play(light.animate.shift(3 * UP), run_time=5)
        # self.play(light.animate.shift(10 * OUT), run_time=5)

        return



# class Vishal(Scene):
#     def construct(self):
#         rect = Rectangle(width=9, height=0.5).shift(DOWN*2).set_color(WHITE).set_opacity(1.0)
#         self.add(rect)
#
#         rect_vert = Rectangle(width=0.5, height=4).shift(LEFT*3).set_color(WHITE).set_opacity(1.0)
#         self.add(rect_vert)
#
#         rect_vert = rect_vert.copy().move_to(ORIGIN)
#         self.add(rect_vert)
#
#         rect_vert = rect_vert.copy().move_to(ORIGIN+3*RIGHT)
#         self.add(rect_vert)
#
#         return


class Vishal2(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 10], y_range=[-1, 8])
        axes.add_coordinate_labels()

        # sin_graph = axes.get_graph(
        #     lambda x: 2 * math.sin(x),
        #     color=BLUE
        # )
        # sin_label = axes.get_graph_label(sin_graph, "\\sin(x)")
        #
        # relu_graph = axes.get_graph(
        #     lambda x: (x if x > 0 else 0),
        #     use_smoothing=False,
        #     color=YELLOW
        # )
        # relu_label = axes.get_graph_label(relu_graph, Text("ReLU"))
        #
        # step_graph = axes.get_graph(
        #     lambda x: (2.0 if x > 3 else 1.0),
        #     discontinuities=[3],
        #     color=GREEN
        # )
        # step_label = axes.get_graph_label(step_graph, Text("Step"))
        #
        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        # self.play(
        #     ShowCreation(sin_graph),
        #     FadeIn(sin_label, RIGHT)
        # )
        # self.wait(2)
        # self.play(
        #     ReplacementTransform(sin_graph, relu_graph),
        #     FadeTransform(sin_label, relu_label)
        # )
        # self.wait(2)
        # self.play(
        #     ReplacementTransform(relu_graph, step_graph),
        #     FadeTransform(relu_label, step_label)
        # )
        # self.wait(2)

        parabola = axes.get_graph(lambda x: 0.25 * x ** 2)
        parabola.set_stroke(BLUE)
        self.play(
            # FadeOut(step_graph),
            # FadeOut(step_label),
            ShowCreation(parabola)
        )
        self.wait()

        dot = Dot(color=RED)
        dot.move_to(axes.i2gp(2, parabola))
        self.play(FadeIn(dot))

        lines = [
            # always_redraw(lambda: Line(dot.get_center(), axes.get_x_axis().n2p(axes.get_x_axis().p2n(dot.get_x())[0]))),
            # always_redraw(lambda: Line(dot.get_center(), axes.get_y_axis().n2p(dot.get_y()) * [0, 1, 0]))
            always_redraw(lambda: axes.get_h_line(dot.get_right())),
            always_redraw(lambda: axes.get_v_line(dot.get_bottom()))
        ]

        print(axes.get_x_axis().p2n(dot.get_x()))

        self.add(*lines)

        dot_1 = Dot(axes.get_x_axis().n2p(2))       # Here 2 is in the axes domain
        self.add(dot_1)

        x_tracker = ValueTracker(2)
        f_always(
            dot.move_to,
            lambda: axes.i2gp(x_tracker.get_value(), parabola)
        )
        self.play(x_tracker.animate.set_value(4), run_time=3)
