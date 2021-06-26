from manimlib import *
from modules.txt import MultiLineText
from modules.ScreenGrid import ScreenGrid
import functools

FONT = "LM Roman 10"

COLOR_XOR_0 = RED
COLOR_XOR_1 = BLUE


class TableXOR(VGroup):
    CONFIG = {
        "cell_width": 1,
        "cell_height": 0.5,
    }

    def __init__(self, n_rows=5, n_cols=3, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        rects = VGroup()
        l_rects = []
        data = [["X", "Y", "XOR"], [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
        for r in range(n_rows):
            for c in range(n_cols):
                # n = r*n_cols + c
                rect = Rectangle(width=self.cell_width, height=self.cell_height)
                l_rects.append(rect)
                fill_color = GREY_B
                text_color = GREY_E
                if r == 0 or c == 2:
                    fill_color = GREY_D
                    text_color = WHITE

                rect.set_fill(fill_color, 1)
                rect.add(Text(str(data[r][c]), font=FONT, color=text_color).scale(0.5))

                rects.add(rect)

        rects.arrange_in_grid(buff=0, n_rows=n_rows, n_cols=n_cols)

        signs = VGroup()
        signs.add(Cross(Dot(rects[5].get_right()), stroke_width=[0, 6, 0]).shift(RIGHT * 0.3))
        signs.add(Dot(rects[8].get_right(), color=COLOR_XOR_1).shift(RIGHT * 0.3))
        signs.add(Dot(rects[11].get_right(), color=COLOR_XOR_1).shift(RIGHT * 0.3))
        signs.add(Cross(Dot(rects[14].get_right()), stroke_width=[0, 6, 0]).shift(RIGHT * 0.3))

        self.rects = rects
        self.signs_xor0 = VGroup(signs[0], signs[3])
        self.signs_xor1 = VGroup(signs[1], signs[2])
        self.enclosure_rectangles()

    def enclosure_rectangles(self):
        rects = self.rects
        VGroup(rects[0], rects[1], rects[2])

        rect0 = Rectangle(color=COLOR_XOR_0).set_stroke(width=3)
        rect0.surround(VGroup(rects[3], rects[4], rects[5]), buff=0, stretch=True).shift(UP * 0.01)
        rect3 = rect0.copy()
        rect3.surround(VGroup(rects[12], rects[13], rects[14]), buff=0, stretch=True).shift(DOWN * 0.01)

        rect1 = rect0.copy().set_color(color=COLOR_XOR_1)
        rect1.surround(VGroup(rects[6], rects[7], rects[8]), buff=0, stretch=True).shift(DOWN * 0.02)
        rect2 = rect1.copy()
        rect2.surround(VGroup(rects[9], rects[10], rects[11]), buff=0, stretch=True).shift(UP * 0.02)

        self.enclosure_xor0 = VGroup(rect0, rect3)
        self.enclosure_xor1 = VGroup(rect1, rect2)


class Neuron(VGroup):
    CONFIG = {
        "edge_color": GREY_B,
        "edge_thickness": 2,
        "arrow_tip_size": 0.01,
        "neuron_radius": 2,
        "neuron_thickness": 5,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
    }

    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        cl = Circle(
            radius=self.neuron_radius,
            stroke_color=GREEN,
            stroke_width=self.neuron_thickness
        ).shift(DOWN)
        self.add(cl)

        cl_center = cl.get_center()

        grp_sq1 = VGroup()
        sq1 = Rectangle(height=1.0, width=1.2).shift(cl_center + LEFT * 1.0).round_corners(0.1)
        arrows_inp = VGroup()
        for i in range(-2, 3, 1):
            ln = Line(
                cl_center + 3 * LEFT + 0.4 * i * DOWN,
                # cl_center-np.array([self.neuron_radius-0.1, 0, 0]),
                sq1.get_left(),
                buff=0.00,
                stroke_color=self.edge_color,
                stroke_width=self.edge_thickness,
            )
            arrows_inp.add(ln)
        self.arrows_inp = arrows_inp

        txt1_1 = Tex(r"z").scale(1.0).move_to(sq1.get_center())
        txt1_2 = Tex(r"\sum\limits_{i} W_i \cdot x_i + b").scale(0.5).move_to(sq1.get_bottom()).shift(DOWN * 0.4)
        grp_sq1.add(sq1, txt1_1, txt1_2)
        self.add(grp_sq1)
        self.neuron_sq1 = sq1
        self.grp_sq1 = grp_sq1

        grp_sq2 = VGroup()
        sq2 = Rectangle(height=1.0, width=1.0).shift(cl_center + RIGHT * 1.1).round_corners(0.1)
        txt2_1 = Tex(r"a").scale(1.0).move_to(sq2.get_center())
        grp_sq2.add(sq2, txt2_1)
        self.add(grp_sq2)
        self.neuron_sq2 = sq2

        grp_z2a = VGroup()
        arrow_z2a = Arrow(sq1.get_right(), sq2.get_left(), buff=0)
        txtz2a_1 = Tex(r"f(z)").move_to(arrow_z2a.get_center()).scale(0.8).shift(UP*0.5)
        grp_z2a.add(arrow_z2a, txtz2a_1)
        self.grp_z2a = grp_z2a
        self.add(grp_z2a)
        self.add(arrows_inp)

        self.grp_to_hide = VGroup(txt1_2, arrows_inp)
        return

    def get_inp_edge_animations(self):
        arrows_inp_copy = self.arrows_inp.copy()
        arrows_inp_copy.set_stroke(
            self.edge_propogation_color,
            width=1.5 * self.edge_thickness
        )
        return [ShowCreationThenDestruction(
            arrows_inp_copy,
            run_time=self.edge_propogation_time,
            lag_ratio=0.5,
        )]


class ReLUUnderstanding(Scene):
    def construct(self):
        self.subText = self.header1 = self.headerline1 = None

        mode = 1
        self.sc_add_heading(mode=mode)
        self.sub("ReLU: Short for Rectified Linear Unit, is one of the most-common activation functions"
                 " across neural networks.")
        self.sc_relu_full_form(mode=mode)

        self.sc_neuron_scene()
        # self.sc_neuron_scene(add=False)
        # self.sc_relu_variants()
        # self.wait(2)
        # self.sc_relu_variants(add=False)
        # self.wait(2)
        # self.embed()
        # self.sc_relu_visualize1()

        # self.sc_relu_solution()
        return

    def sc_add_heading(self, mode=0):
        if mode in [0, 1]:
            self.header1 = Text("ReLU", font=FONT).scale(1).shift(UP * 3.3).set_stroke(BLACK, 1, background=True)
            self.headerline = Line(LEFT * 5.5, RIGHT * 5.5).set_fill(BLACK, 1).align_to(self.header1, BOTTOM).shift(
                DOWN * 0.3)
            self.header2 = Text("[Rectified Linear Unit]", font=FONT).scale(0.4). \
                align_to(self.headerline.get_end(), RIGHT).align_to(self.header1, BOTTOM).shift(DOWN * 0.13)

            if mode == 0:
                self.play(Write(self.header1, run_time=2), GrowFromEdge(self.headerline, LEFT, run_time=2, buff=0.0))
            elif mode == 1:
                self.add(self.header1, self.headerline)
        return

    def sc_relu_full_form(self, mode=0):
        if mode == 0:
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
            self.wait(1)
        elif mode == 1:
            self.add(self.header2)
        elif mode == -1:
            self.remove(self.relu_full_form)

    def sc_neuron_scene(self, add=True):
        if add:
            # grp_neuron = VGroup()
            neuron = Neuron().shift(UP)
            self.play(Write(neuron))
            # grp_neuron.add(neuron)

            anims_inp = neuron.get_inp_edge_animations()
            self.play(*anims_inp, run_time=2)

            x_range = [-4, 4]
            density = 1 / 2
            color_axes = GREY_D
            color_graph = BLUE_B
            axes = Axes((-4, 5), (-1, 5),
                        width=9 * density, height=6 * density,
                        x_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes},
                        y_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes}
                        ).shift(RIGHT*3+UP * 0.8)

            relu_equation = Tex(r"""
                            f(z) = max(z, 0)
                            \\
                            f(z) = \begin{cases}z & z \geq 0\\0 & z < 0\end{cases}
                            """).shift(3.5 * RIGHT + 2.4 * DOWN).scale(0.9)
            relu_graph = axes.get_graph(lambda x: max(x, 0), color=color_graph, stroke_width=3.0, x_range=x_range,
                                        **{"discontinuities": 0})
            self.play(FadeIn(relu_equation, LEFT_SIDE), neuron.animate.shift(2.0*LEFT), run_time=2)
            self.play(FadeOut(neuron.grp_to_hide), run_time=0.5)
            neuron.grp_sq1.remove(neuron.grp_sq1[-1])
            neuron.remove(neuron.arrows_inp)
            self.play(neuron.animate.shift(0.8*LEFT), Write(axes, lag_ratio=0.01, run_time=1))
            self.play(ShowCreation(relu_graph))

            x_tracker = ValueTracker(4)

            dot = Dot(color=RED)
            dot.move_to(axes.i2gp(x_tracker.get_value(), relu_graph))
            self.play(FadeIn(dot, scale=0.5))

            decimal_inp = DecimalNumber(
                x_tracker.get_value(),
                num_decimal_places=2,
                font=FONT
            ).scale(0.5).move_to(neuron.neuron_sq1.get_bottom()).shift(0.5*DOWN)
            decimal_out = DecimalNumber(
                relu_graph.underlying_function(x_tracker.get_value()),
                num_decimal_places=2,
                font=FONT
            ).scale(0.5).move_to(neuron.neuron_sq2.get_bottom()).shift(0.5*DOWN)

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
            self.play(x_tracker.animate.set_value(-4), run_time=1)
            self.play(x_tracker.animate.set_value(4), run_time=1)

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
        self.embed()
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
        screen_grid = ScreenGrid(width=2, height=2, rows=4, columns=4, show_numbers=False, grid_opacity=0.4) \
            # .to_edge(BOTTOM + RIGHT,buff=0.1)
        self.add(screen_grid)

        # pos_table = 5*LEFT
        # table = TableXOR()
        # self.sub("We will take the classic example of learning the XOR using neural networks.")
        # self.play(Write(table.rects.shift(pos_table)))
        # self.play(
        #     FadeIn(table.signs_xor0.shift(pos_table), run_time=0.3),
        #     FadeIn(table.enclosure_xor0.shift(pos_table), run_time=0.3)
        # )
        # self.play(
        #     FadeIn(table.signs_xor1.shift(pos_table), run_time=0.3),
        #     FadeIn(table.enclosure_xor1.shift(pos_table), run_time=0.3)
        # )

        axes = Axes(
            (-1, 1), (-1, 1),
            width=2, height=2,
            axis_config={"include_tip": False},
            x_axis_config={"xtra_pad_start": 0.0, "xtra_pad_end": 0.2},
            y_axis_config={"xtra_pad_start": 0.0, "xtra_pad_end": 0.2}
        )
        self.add(axes)

        dots1 = VGroup(
            Cross(Dot(axes.c2p(*[0, 0, 0])), stroke_width=[0, 6, 0]),
            Cross(Dot(axes.c2p(*[1, 1, 0])), stroke_width=[0, 6, 0])
        ).set_color(COLOR_XOR_0)
        dots2 = VGroup(
            Dot(axes.c2p(*[0, 1, 0])),
            Dot(axes.c2p(*[1, 0, 0]))
        ).set_color(COLOR_XOR_1)
        self.add(dots1)
        self.add(dots2)
        # self.sub("The same is depicted on the graph.")
        # self.sub("When x and y are the same, the output is zero (depicted by a x)."
        #          "Whereas, when one is 0, and the other is 1, we get 1 as output. (depicted by a blue dot)")
        # self.sub("No matter what we do, we cannot pass a line through this graph to segment out the cross from the dot")
        #
        # l_values = [
        #     (PI, [0.5, 0.5, 0]),
        #     (3*PI/4, [0.5, 0.5, 0]),
        #     (PI/2, [0.5, 0.5, 0]),
        #     (PI/4, [0.5, 0.5, 0]),
        #     (0, [0.5, 0.5, 0]),
        #     (0, [0.5, 0, 0]),
        #     (0, [0.5, 1, 0]),
        # ]
        #
        # ln = DashedLine(0.8*LEFT, 0.8*RIGHT).set_angle(l_values[0][0]).move_to(axes.c2p(*l_values[0][1]))
        # ln.save_state()
        #
        # def update_pos(obj, alpha):
        #     print("i= {}".format(i))
        #     ln.restore()
        #     diff_angle = l_values[i][0] - l_values[i-1][0]
        #     diff_center = np.array(l_values[i][1]) - np.array(l_values[i-1][1])
        #     ln.rotate(alpha*diff_angle)
        #     ln.shift(alpha*diff_center)
        #     if alpha == 1:
        #         ln.save_state()
        #
        # self.add(ln)
        # for i, [slope, center] in enumerate(l_values):
        #     if i == 0:
        #         continue
        #     # self.play(ln.animate.set_angle(slope).move_to(axes.c2p(*center)))
        #     # ln.set_angle(slope)
        #     # ln.move_to(axes.c2p(*center))
        #     self.play(UpdateFromAlphaFunc(ln, update_pos, run_time=0.7))
        #     self.wait(1)
        #
        # self.wait(2)
        # self.play(FadeOut(ln))
        # self.remove(ln)

        # show a line with different orientations...
        # No amount of squishing of the axes will again solve it for us, since linear transformation leave lines as lines.
        # And the line connecting the cross, and dots criss-cross with one another.

        axes.save_state()
        screen_grid.save_state()

        l_targets = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[1, 1/2, 0], [1/2, 1, 0], [0, 0, 0]]),
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            np.array([[1, -1/2, 0], [-1/2, 1, 0], [0, 0, 0]]),
            np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]]),
        ]

        def f1(obj):
            obj[0].move_to(axes.c2p(*[0, 0, 0]))
            obj[1].move_to(axes.c2p(*[1, 1, 0]))

        def f2(obj):
            obj[0].move_to(axes.c2p(*[0, 1, 0]))
            obj[1].move_to(axes.c2p(*[1, 0, 0]))

        dots1.add_updater(f1)
        dots2.add_updater(f2)

        # self.remove(screen_grid)
        self.wait(2)

        # def transform_fn(pt, affine):
        #     pt2 = np.matmul(affine, pt)
        #     return pt2
        #
        # def update_affine(obj, alpha):
        #     print("i= {}".format(i))
        #     axes.restore()
        #     diff_affine = alpha*(l_targets[i] - l_targets[i - 1])
        #     axes.apply_function(functools.partial(transform_fn, affine=diff_affine))
        #     if alpha == 1:
        #         axes.save_state()
        #
        # for i, target_mat in enumerate(l_targets):
        #     if i == 0:
        #         continue
        #     self.play(UpdateFromAlphaFunc(axes, update_affine, run_time=0.7))
        #     self.wait(1)

        # def transform_fn(pt):
        #     pt2 = np.matmul(l_targets[1], pt)
        #     return pt2
        # self.play(axes.animate.apply_function(transform_fn), run_time=4)

        def transform_fn(pt):
            pt2 = np.matmul(l_targets[3], pt)
            return pt2
        self.play(axes.animate.apply_function(transform_fn), screen_grid.animate.apply_function(transform_fn), run_time=4)

        axes.restore()
        screen_grid.restore()

        return

    def sc_relu_solution(self, add=True):

        screen_grid = ScreenGrid(width=2, height=2, rows=4, columns=4, show_numbers=False, grid_opacity=0.4)
        self.add(screen_grid)

        axes = Axes(
            (-1, 1), (-1, 1),
            width=2, height=2,
            axis_config={"include_tip": False},
            x_axis_config={"xtra_pad_start": 0.0, "xtra_pad_end": 0.2},
            y_axis_config={"xtra_pad_start": 0.0, "xtra_pad_end": 0.2}
        )
        self.add(axes)

        axes2 = Axes(
            (-1, 1), (-1, 1),
            width=2, height=2,
            axis_config={},
            x_axis_config={"include_tip": False, "include_ticks": False,
                           "xtra_pad_start": 0.0, "xtra_pad_end": 0.2, "stroke_width": 0},
            y_axis_config={"include_tip": False, "include_ticks": False,
                           "xtra_pad_start": 0.0, "xtra_pad_end": 0.2, "stroke_width": 0}
        )
        axes3 = Axes(
            (-1, 1), (-1, 1),
            width=2, height=2,
            axis_config={},
            x_axis_config={"include_tip": False, "include_ticks": False,
                           "xtra_pad_start": 0.0, "xtra_pad_end": 0.2, "stroke_width": 0},
            y_axis_config={"include_tip": False, "include_ticks": False,
                           "xtra_pad_start": 0.0, "xtra_pad_end": 0.2, "stroke_width": 0}
        )

        dots1 = VGroup(
            Cross(Dot(axes.c2p(*[0, 0, 0])), stroke_width=[0, 6, 0]),
            Cross(Dot(axes.c2p(*[1, 1, 0])), stroke_width=[0, 6, 0])
        ).set_color(COLOR_XOR_0)
        dots2 = VGroup(
            Dot(axes.c2p(*[0, 1, 0])),
            Dot(axes.c2p(*[1, 0, 0]))
        ).set_color(COLOR_XOR_1)
        self.add(dots1)
        self.add(dots2)

        # axes2 = axes.copy()
        # self.remove(axes)
        # self.add(axes2)

        def f1(obj):
            obj[0].move_to(axes2.c2p(*[0, 0, 0]))
            obj[1].move_to(axes2.c2p(*[1, 1, 0]))

        def f2(obj):
            obj[0].move_to(axes2.c2p(*[0, 1, 0]))
            obj[1].move_to(axes2.c2p(*[1, 0, 0]))

        dots1.add_updater(f1)
        dots2.add_updater(f2)

        # grp = VGroup(screen_grid, axes, dots1, dots2)
        # grp.shift(3*LEFT)

        self.wait(2)

        target_mat = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])/2
        def transform_fn(pt):
            pt2 = np.matmul(target_mat, pt)
            return pt2
        self.play(axes2.animate.apply_function(transform_fn), run_time=4)
        self.remove(axes2)

        dots1.clear_updaters()
        dots2.clear_updaters()

        init_pt00 = axes3.p2c(dots1[0].get_center())
        init_pt01 = axes3.p2c(dots1[1].get_center())

        init_pt10 = axes3.p2c(dots2[0].get_center())
        init_pt11 = axes3.p2c(dots2[1].get_center())

        def f1(obj):
            obj[0].move_to(axes3.c2p(*init_pt00))
            obj[1].move_to(axes3.c2p(*init_pt01))
        def f2(obj):
            obj[0].move_to(axes3.c2p(*init_pt10))
            obj[1].move_to(axes3.c2p(*init_pt11))

        dots1.add_updater(f1)
        dots2.add_updater(f2)
        self.wait(2)

        target_translation = np.array([0, -1, 0])/2
        def transform_fn(pt):
            pt2 = pt + target_translation
            return pt2
        self.play(axes3.animate.apply_function(transform_fn), run_time=4)
        self.remove(axes3)

        dots1.clear_updaters()
        dots2.clear_updaters()

        def twoD2threeD(pt):
            return np.pad(np.array(pt), [0, 1])

        init_pt00 = twoD2threeD(axes.p2c(dots1[0].get_center()))
        init_pt01 = twoD2threeD(axes.p2c(dots1[1].get_center()))

        init_pt10 = twoD2threeD(axes.p2c(dots2[0].get_center()))
        init_pt11 = twoD2threeD(axes.p2c(dots2[1].get_center()))

        self.play(dots1[0].animate.move_to(twoD2threeD(axes.p2c(np.array([0, 0, 0])))))

        # Transform the four dots by the matrix
        ln = DashedLine(LEFT*1.5, RIGHT*1.5)\
                .set_angle(Line(dots1[0].get_center(), dots1[1].get_center()).get_slope()) \
                .move_to(twoD2threeD(axes.p2c(np.array([0.5, 0.125, 0]))))
        self.add(ln)
        self.wait(2)
        return

    def sub(self, txt):
        return
        subText = Text(txt, font=FONT).scale(0.6).to_corner(BOTTOM, buff=0.01)
        if self.subText:
            self.play(ReplacementTransform(self.subText, subText, run_time=0.01))
        else:
            self.play(Write(subText, run_time=0.01))
        self.subText = subText
