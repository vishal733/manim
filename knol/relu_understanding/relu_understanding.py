from manimlib import *
from modules.txt import MultiLineText
from modules.ScreenGrid import ScreenGrid


class Neuron(VGroup):
    CONFIG = {
        "edge_color": GREY_B,
        "edge_thickness": 0.01,
        "arrow_tip_size": 0.01
    }

    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        radius = 0.5
        cl = Circle(
            radius=0.5,
            stroke_color=GREEN,
            stroke_width=1
        )
        self.add(cl)

        for i in range(-2, 3, 1):
            arrow = Arrow(
                3 * LEFT + 0.7 * i * DOWN, cl.get_center(),
                buff=radius + 0.1,
                fill_color=self.edge_color,
                # thickness=self.edge_thickness,
            ).set_stroke(width=self.edge_thickness)
            self.add(arrow)

        arrow = Arrow(
            cl.get_center(), 2 * RIGHT,
            buff=radius + 0.1,
            fill_color=self.edge_color,
            # thickness=self.edge_thickness,
        ).set_stroke(width=self.edge_thickness)
        self.add(arrow)
        return


FONT = "LM Roman 10"


class ReLUUnderstanding(Scene):
    def construct(self):
        self.subText = self.header1 = self.headerline1 = None

        # self.sc_add_heading()
        # self.sub("ReLU: Short for Rectified Linear Unit, is one of the most-common activation functions"
        #          " across neural networks.")
        # self.sc_relu_full_form()
        # self.sc_neuron_scene()
        # self.sc_neuron_scene(add=False)
        # self.sc_relu_variants()
        # self.wait(2)
        # self.sc_relu_variants(add=False)
        # self.wait(2)
        # self.embed()
        self.sc_relu_visualize1()
        return

    def sc_add_heading(self):
        self.header1 = Text("ReLU", font=FONT).scale(1).shift(UP * 3.3).set_stroke(BLACK, 1, background=True)
        self.headerline = Line(LEFT * 5.5, RIGHT * 5.5).set_fill(BLACK, 1).align_to(self.header1, BOTTOM).shift(
            DOWN * 0.3)

        self.play(Write(self.header1, run_time=2), GrowFromEdge(self.headerline, LEFT, run_time=2, buff=0.0))
        self.header2 = Text("[Rectified Linear Unit]", font=FONT).scale(0.4). \
            align_to(self.headerline.get_end(), RIGHT).align_to(self.header1, BOTTOM)
        return

    def sc_relu_full_form(self, add=True):
        if add:
            color_1, color_2, color_3 = (BLUE_E, GREEN, YELLOW)
            self.relu_full_form = Text(
                "Rectified Linear Unit", font=FONT,
                t2c={"Rectified": color_1, "Linear": color_2, "Unit": color_3}
            ).scale(0.8).shift(UP * 2.3)
            header_copy = self.header1.copy()
            header_copy.set_color_by_t2c({"Re": color_1, "L": color_2, "U": color_3})
            self.play(Write(self.relu_full_form), Write(header_copy, run_time=2))
            self.wait(2)
            self.play(ReplacementTransform(self.relu_full_form, self.header2), FadeOut(header_copy))
            self.remove(header_copy)
            self.remove()
        else:
            self.remove(self.relu_full_form)

    def sc_neuron_scene(self, add=True):
        if add:
            neuron = Neuron().shift(UP)
            self.add(neuron)

            x_range = [-4, 4]
            density = 1 / 2
            color_axes = GREY_D
            color_graph = BLUE_B
            axes = Axes((-4, 5), (-1, 5),
                        width=9 * density, height=6 * density,
                        x_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes},
                        y_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes}
                        ).shift(DOWN * 1.5)

            relu_equation = Tex(r"""
                            f(x) = max(x, 0)
                            \\
                            f(x) = \begin{cases}x & x \geq 0\\0 & x < 0\end{cases}
                            """).shift(4.5 * RIGHT + 2 * DOWN)
            relu_graph = axes.get_graph(lambda x: max(x, 0), color=color_graph, stroke_width=3.0, x_range=x_range,
                                        **{"discontinuities": 0})
            self.play(Write(relu_equation))
            self.play(Write(axes, lag_ratio=0.01, run_time=1))
            self.play(ShowCreation(relu_graph))

            x_tracker = ValueTracker(4)

            dot = Dot(color=RED)
            dot.move_to(axes.i2gp(x_tracker.get_value(), relu_graph))
            self.play(FadeIn(dot, scale=0.5))

            decimal_inp = DecimalNumber(
                x_tracker.get_value(),
                num_decimal_places=2
            ).scale(0.5).shift(3 * LEFT)
            decimal_out = DecimalNumber(
                relu_graph.underlying_function(x_tracker.get_value()),
                num_decimal_places=2
            ).scale(0.5).shift(3 * RIGHT)

            def updater(obj):
                vall = x_tracker.get_value()
                obj.set_value(vall)

            decimal_inp.add_updater(updater)
            decimal_out.add_updater(lambda obj: obj.set_value(relu_graph.underlying_function(x_tracker.get_value())))

            # f_always(label1.set)
            f_always(dot.move_to, lambda: axes.i2gp(x_tracker.get_value(), relu_graph))
            h_line = always_redraw(lambda: axes.get_h_line(dot.get_left()))
            v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom()))
            self.add(h_line, v_line)
            self.play(FadeIn(decimal_inp), FadeIn(decimal_out))
            self.play(x_tracker.animate.set_value(-4), run_time=3)
            self.play(x_tracker.animate.set_value(4), run_time=3)

            self.neuron = neuron
            self.axes1 = axes
            self.relu_graph1 = relu_graph
            self.decimal_inp = decimal_inp
            self.decimal_out = decimal_out
            self.h_line = h_line
            self.v_line = v_line
            self.dot1 = dot
            self.relu_equation = relu_equation
        else:
            self.remove(self.neuron, self.axes1, self.relu_equation, self.relu_graph1, self.dot1)
            self.remove(self.decimal_inp, self.decimal_out, self.h_line, self.v_line)

        return

    def sc_relu_variants(self, add=True):
        if add:
            txt1 = Text(
                "Scene: ReLU Variants"
            ).shift(UP)
            self.add(txt1)
            self.vgrp_relu_variants = VGroup(txt1)
            self.sub(
                "There are different variants of ReLU, with slight variation in the output: Leaky, Parameterized, etc."
                "I won't be going into the advantages of these. You can find it at numerous places. Goal is to visualize."
            )
        else:
            self.remove(self.vgrp_relu_variants)
            self.sub("")
        return

    def sc_relu_visualize1(self, add=True):
        # To solve this, we will start with a classic problem of teaching XOR function to Neural Network
        screen_grid = ScreenGrid(width=2, height=2, rows=4, columns=4, show_numbers=False).to_edge(BOTTOM + RIGHT,
                                                                                                   buff=0.1)
        self.add(screen_grid)

        axes = Axes((-1, 1), (-1, 1), width=2, height=2, axis_config={"include_tip": False})
        self.add(axes)

        dots1 = VGroup(
            Cross(Dot(axes.c2p(*[0, 0, 0]))),
            Cross(Dot(axes.c2p(*[1, 1, 0])))
        ).set_color(RED)
        dots2 = VGroup(
            Dot(axes.c2p(*[0, 1, 0])),
            Dot(axes.c2p(*[1, 0, 0]))
        ).set_color(BLUE)
        self.add(dots1)
        self.add(dots2)

        axes2 = axes.copy()
        self.add(axes2)
        def f1(obj):
            obj[0].move_to(axes.c2p(*[0, 0, 0]))
            obj[1].move_to(axes.c2p(*[1, 1, 0]))

        def f2(obj):
            obj[0].move_to(axes.c2p(*[0, 1, 0]))
            obj[1].move_to(axes.c2p(*[1, 0, 0]))

        dots1.add_updater(f1)
        dots2.add_updater(f2)

        def transform_fn(pt):
            pt2 = np.matmul(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])/np.sqrt(1), pt)
            return pt2
        self.play(axes.animate.apply_function(transform_fn), run_time=4)
        self.add(dots1)
        self.add(dots2)

        return

    def sub(self, txt):
        subText = Text(txt, font=FONT).scale(0.35).to_corner(BOTTOM, buff=0.01)
        if self.subText:
            self.play(ReplacementTransform(self.subText, subText, run_time=0.01))
        else:
            self.play(Write(subText, run_time=0.01))
        self.subText = subText
