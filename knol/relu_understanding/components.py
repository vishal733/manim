from manimlib import *
from modules.txt import MultiLineText
from modules.ScreenGrid import ScreenGrid
import functools
from knol.relu_understanding.neural import Neuron, NetworkMobject

FONT = "LM Roman 10"

COLOR_XOR_0 = RED
COLOR_XOR_1 = BLUE


def get_relu_equation():
    eq1 = Tex(r"""
               f(z) = max(z, 0)
               \\
               f(z) = \begin{cases}z & z \geq 0\\0 & z < 0\end{cases}
               """).shift(3.5 * RIGHT + 2.4 * DOWN).scale(0.9)
    eq2 = Tex(r"""
               f(z) = max(z, 0)
               \\
               f(z) = \begin{cases}z & z \geq 0\\0 & z < 0\end{cases}
               """).shift(3.5 * RIGHT + 2.4 * DOWN).scale(0.9)
    eq2.set_color_by_tex_to_color_map({
        "z & z \geq 0": YELLOW
    })
    eq3 = Tex(r"""
               f(z) = max(z, 0)
               \\
               f(z) = \begin{cases}z & z \geq 0\\0 & z < 0\end{cases}
               """).shift(3.5 * RIGHT + 2.4 * DOWN).scale(0.9)
    eq3.set_color_by_tex_to_color_map({
        "0 & z < 0": YELLOW
    })
    sq = Square()
    sq.surround(eq1, stretch=True)
    return (eq1, eq2, eq3, sq)


class TableReLU(VGroup):
    CONFIG = {
        "scale_val": "1.0"
    }
    def __init__(self, labels=['z', 'a'], *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)

        self.labels = ['z', 'a']
        vert_line = Line(UP, DOWN, buff=0).scale(self.scale_val)
        vert_line_0 = Line(UP, ORIGIN, buff=0).scale(self.scale_val, about_point=ORIGIN)
        vert_line_1 = Line(ORIGIN, DOWN, buff=0).scale(self.scale_val, about_point=ORIGIN)
        horz_line = Line(LEFT*2, RIGHT*2, buff=0).scale(self.scale_val)
        self.add(vert_line)
        self.add(vert_line_0)
        self.add(vert_line_1)
        self.add(horz_line)
        self.lines = []
        self.lines.append(vert_line_0)
        self.lines.append(vert_line_1)

        dot = Dot(ORIGIN)
        self.origin = dot

        self.locs = []

        dot_0 = Dot().set_fill(opacity=0.0).\
            shift((horz_line.get_start()-dot.get_center())/2).\
            shift((vert_line.get_start()-dot.get_center())/2)
        dot_1 = Dot().set_fill(opacity=0.0). \
            shift((horz_line.get_end() - dot.get_center()) / 2). \
            shift((vert_line.get_start() - dot.get_center()) / 2)
        self.add(dot_0, dot_1)
        self.locs.append(VGroup(dot_0, dot_1))

        dot_0 = Dot().set_fill(opacity=0.0). \
            shift((horz_line.get_start() - dot.get_center()) / 2). \
            shift((vert_line.get_end() - dot.get_center()) / 2)
        dot_1 = Dot().set_fill(opacity=0.0). \
            shift((horz_line.get_end() - dot.get_center()) / 2). \
            shift((vert_line.get_end() - dot.get_center()) / 2)
        self.add(dot_0, dot_1)
        self.locs.append(VGroup(dot_0, dot_1))

        self.start_loc = 0
        self.curr_count = 0

    def add_pt_if_not_exists(self, index):
        if index < len(self.locs):
            return
        if index > len(self.locs)+1:
            raise Exception()
        vect = [
            self.locs[1][0].get_center() - self.locs[0][0].get_center(),
            self.locs[1][1].get_center() - self.locs[0][1].get_center()
        ]
        dot_0 = Dot().set_fill(opacity=0.0).move_to(self.locs[-1][0]).shift(vect[0])
        dot_1 = Dot().set_fill(opacity=0.0).move_to(self.locs[-1][1]).shift(vect[1])
        self.locs.append(VGroup(dot_0, dot_1))
        return

    def get_next_segment_animation(self, l_objs):
        anims = []
        if self.curr_count == 0:
            txt0 = Text(self.labels[0], font=FONT).move_to(self.locs[0][0])
            txt1 = Text(self.labels[1], font=FONT).move_to(self.locs[0][1])
            anims.extend([FadeIn(txt0), FadeIn(txt1)])
        self.curr_count += 1
        self.add_pt_if_not_exists(self.curr_count)

        if self.curr_count >= len(self.lines):
            last_line = self.lines[-1].copy()
            last_line.shift(last_line.get_end()-last_line.get_start())
            anims.append(GrowFromPoint(last_line, last_line.get_start()))
            self.lines.append(last_line)

        path = ArcBetweenPoints(l_objs[0].get_center(), self.locs[self.curr_count][0].get_start())
        anims.append(MoveAlongPath(l_objs[0], path))
        path = ArcBetweenPoints(l_objs[1].get_center(), self.locs[self.curr_count][1].get_start(), angle=-TAU/4)
        anims.append(MoveAlongPath(l_objs[1], path))

        return anims


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


class SceneComponents(Scene):

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
            color_1, color_2, color_3 = (BLUE_C, GREEN_D, YELLOW)
            self.relu_full_form = Text(
                "Rectified Linear Unit", font=FONT,
                t2c={"Rectified": color_1, "Linear": color_2, "Unit": color_3}
            ).scale(1.0).shift(UP * 2.2)
            header_copy = self.header1.copy()
            header_copy.set_color_by_t2c({"Re": color_1, "L": color_2, "U": color_3})
            self.play(Write(self.relu_full_form), Write(header_copy, run_time=2))
            self.wait()
            self.play(ReplacementTransform(self.relu_full_form, self.header2), FadeOut(header_copy))
            self.remove(header_copy)
            self.remove()
            self.wait()

            txt1 = Text("A common activation function used in neural networks.",
                        font=FONT,
                        t2c={"activation function": color_1, "neural networks": color_1})
            txt2 = Text("PRE-REQUISITES:\n  Basic familiarity with neural networks.",
                        font=FONT, lsh=1,
                        t2c={"activation function": color_1, "neural networks": color_1})\
                .align_to(txt1, LEFT).shift(DOWN)
            txt3 = Text("GOAL:\n  Visualize effect of ReLU using an example.",
                        font=FONT, lsh=1,
                        t2c={"Visualize": color_1, "ReLU": color_1})\
                .align_to(txt1, LEFT).shift(DOWN*2)
            self.to_remove = VGroup(txt1, txt2, txt3)
            self.play(Write(txt1))
            self.wait()
            self.play(txt1.animate.shift(UP*2), Write(txt2))
            self.wait()
            self.play(txt2.animate.shift(UP))
            self.wait()
            self.play(Write(txt3))
            # self.add(txt1)
            self.wait()
            self.play(FadeOut(self.to_remove))
            self.wait()
        elif mode == 1:
            self.add(self.header2)
        elif mode == -1:
            self.remove(self.relu_full_form)
            self.remove(self.to_remove)

    def sc_neuron_scene(self, add=True):
        if add:
            layer_sizes = [3, 1]
            network_config = {
                "neuron_radius": 0.3,
                "layer_specific_radius": {0: 0.75, 1: 2},
                "neuron_stroke_color": BLUE,
                "layer_labels": ["x", None]
            }

            # Create initial circle representing a neuron
            self.network_mob = None
            (network, network_mob) = NetworkMobject.add_network(layer_sizes, None, network_config)
            (self.network, self.network_mob) = (network, network_mob)
            self.network_mob.shift(LEFT)
            self.network_mob.set_label_layer("x", 0)

            neuron_1 = self.network_mob.get_neuron(1, 0)
            cl = Circle(radius=network_config["neuron_radius"]*2)\
                .set_stroke(color=network_config["neuron_stroke_color"])
            self.play(Write(cl))
            # Transform circle into full network
            self.play(LaggedStart(cl.animate.move_to(neuron_1), Write(self.network_mob), lag_ratio=0.3))
            self.remove(cl)

            # Add arrow at end of tiny neuron
            neuron, l_extras0 = self.network_mob.add_within_tiny_neuron(1, 0)
            self.play(Write(l_extras0[-1]))

            self.network_mob.update_edges()
            self.wait(1)
            # return

            # self.feed_forward(np.array([[0, 0, 0]]))

            self.network_mob.save_state()

            # Section: Transform small circle to bigger circle
            l_extras0[-1].add_updater(lambda obj: obj.next_to(neuron, RIGHT, buff=0))
            def modify(obj, alpha):
                obj.restore()
                obj.shift(alpha * LEFT)
                obj.scaleAndShiftNeuronsInLayer(1, scale_factor=1 + alpha * 3/2, shift_factor=alpha * RIGHT)
                obj.update_edges()
                return obj
            self.play(UpdateFromAlphaFunc(self.network_mob, modify))
            l_extras0[-1].clear_updaters()

            self.wait(2)

            neuron, l_extras = self.network_mob.add_within_neuron(1, 0)

            # Start by adding summation symbol inside zoomed neuron
            self.network_mob.save_state()
            def modify(obj, alpha):
                obj.restore()
                neuron.set_transition_alpha(alpha)
                obj.update_edges()
                return obj
            self.play(Write(l_extras[0]), UpdateFromAlphaFunc(self.network_mob, modify))

            txt1_2 = Tex(r"z = \sum\limits_{i} W_i \cdot x_i + b").scale(0.8).move_to(
                self.network_mob.get_bottom()).shift(DOWN * 0.4).move_to(neuron.get_bottom() + 0.8*DOWN)

            # Show remaining flow inside neuron: z, f and a
            self.play(Write(l_extras[1])); self.wait(1)
            self.play(Write(txt1_2)); self.wait(3)
            self.play(FadeOut(txt1_2)); self.remove(txt1_2); self.wait(0.5)
            self.play(Write(l_extras[2])); self.wait(0.5)
            self.play(LaggedStart(FadeOut(l_extras0[-1]), Write(l_extras[3]), lag_ratio=0.5)); self.wait(0.5)
            self.remove(l_extras0[-1])
            bkg_rect = SurroundingRectangle(VGroup(*l_extras[1:]), fill_opacity=0, buff=0.4,
                                            stroke_color=BLUE, stroke_opacity=0.8).round_corners(0.1).shift(LEFT*0.2)
            self.play(FadeOut(self.network_mob), FadeOut(l_extras[0]), Write(bkg_rect))
            self.remove(l_extras[0])
            path = ArcBetweenPoints(l_extras[1][1].get_center(), l_extras[0][0].get_start() - np.array([0.3, 0, 0]))

            txt = Text("Activation Function", font=FONT, color=WHITE).\
                scale(0.8).move_to(bkg_rect.get_top(), BOTTOM).\
                align_to(bkg_rect.get_right(), RIGHT).shift(UP*0.15+LEFT*0.05)
            self.play(MoveAlongPath(l_extras[1][1], path), Write(txt))
            self.wait(2)

            def get_axis_relugraph_pair(remove_ticks=False, stroke_width=3.0, primary=True):
                x_range = [-4, 4]
                density = 1 / 2
                color_axes = GREY_D
                color_graph = BLUE_B
                axes = Axes((-4, 5), (-1, 5),
                            width=9 * density, height=6 * density,
                            x_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes},
                            y_axis_config={"include_tip": True, "xtra_pad_start": 0.2, "color": color_axes}
                            ).shift(RIGHT*3+UP * 0.8)

                relu_graph = axes.get_graph(lambda x: max(x, 0), color=color_graph, stroke_width=stroke_width,
                                            x_range=x_range,
                                            **{"discontinuities": 0})
                if primary:
                    axes.add(Text("z", font=FONT).move_to(axes.x_axis.get_end()).shift(UP*0.3).set_color(color_axes).scale(0.5))
                    axes.add(Text("a", font=FONT).move_to(axes.y_axis.get_end()).shift(RIGHT * 0.3).set_color(color_axes).scale(0.5))
                    # fz_label = axes.get_graph_label(relu_graph, "f(z)")
                    # relu_graph.add(fz_label)
                if remove_ticks:
                    axes.x_axis.remove_ticks()
                    axes.y_axis.remove_ticks()
                return axes, relu_graph

            (axes, relu_graph) = get_axis_relugraph_pair()
            (axes2, relu_graph2) = get_axis_relugraph_pair(remove_ticks=True, primary=False)
            axes_clone = VGroup(axes2, relu_graph2).copy()
            eq1, eq2, eq3, sq1 = get_relu_equation()
            relu_equation = eq1
            # self.add(sq1)
            # self.play(LaggedStart(
            #     FlashAround(sq1, time_width=3, taper_width=3),
            #     FlashAround(sq1, time_width=3, taper_width=3),
            #     lag_ratio=0.7
            # ))
            # self.add(sq1)

            new_vgroup = VGroup(*l_extras[1:], bkg_rect)

            self.play(
                new_vgroup.animate.shift(2 * LEFT),
                txt.animate.shift(2 * LEFT),
                run_time=2)
            self.wait(1)

            # Requirements from activation function
            txt1 = Text("- Non-linearity", font=FONT).scale(0.6).move_to(new_vgroup.get_left(), LEFT).shift(DOWN*1.5)
            txt2 = Text("- First order-derivative defined\n  at every point", font=FONT, lsh=0.7).scale(0.6).move_to(txt1.get_left(), LEFT).shift(DOWN*0.7)

            self.add(txt1)
            self.wait(3)
            self.add(txt2)
            self.wait(3)

            self.play(
                FadeIn(relu_equation, LEFT_SIDE),
                FadeIn(axes, LEFT_SIDE),
                new_vgroup.animate.shift(2.5 * LEFT),
                VGroup(txt1, txt2).animate.shift(2.5 * LEFT),
                txt.animate.shift(2.5 * LEFT),
                # Write(axes, lag_ratio=0.01, run_time=1),
                run_time=2)

            self.wait(1)
            # self.play(FadeOut(txt), run_time=0.1)
            # self.play(Write(axes, lag_ratio=0.01, run_time=1))
            self.wait(2)
            self.play(ShowCreation(relu_graph))

            ln = Line(axes.c2p(*[0, 0, 0]), axes.c2p(*[4, 4, 0])).set_stroke(color=YELLOW, width=8)
            self.play(Write(ln))
            self.wait(2)
            self.remove(ln)
            self.wait()

            ln = Line(axes.c2p(*[-4, 0, 0]), axes.c2p(*[0, 0, 0])).set_stroke(color=YELLOW, width=8)
            self.play(Write(ln))
            self.wait(2)
            self.remove(ln)
            self.wait()
            # l_comps_eq = NetworkMobject.get_neuron_internal_contents(shift=2.75 * LEFT)
            # self.play(LaggedStart(*[Write(e) for e in l_comps_eq]))

            # Replace the
            txt1_clone = txt1.copy().set_stroke(color=GREEN_B).set_fill(color=GREEN)
            self.play(ReplacementTransform(txt1, txt1_clone))
            self.wait()

            self.play(FadeOut(txt), run_time=0.1)
            self.play(new_vgroup.animate.shift(UP * 1.5))

            x_tracker = ValueTracker(4)
            dot = Dot(color=RED)
            dot.move_to(axes.i2gp(x_tracker.get_value(), relu_graph))
            self.play(FadeIn(dot, scale=0.5))
            self.remove(bkg_rect)

            decimal_inp = DecimalNumber(
                x_tracker.get_value(),
                num_decimal_places=2,
                font=FONT,
                edge_to_fix=RIGHT
            ).scale(0.5).move_to(l_extras[1][1].get_center()).shift(LEFT*0.05)
            decimal_out = DecimalNumber(
                relu_graph.underlying_function(x_tracker.get_value()),
                num_decimal_places=2,
                font=FONT
            ).scale(0.5).move_to(l_extras[3][1].get_center()).shift(RIGHT*0.05)
            # f_always(label1.set)
            f_always(dot.move_to, lambda: axes.i2gp(x_tracker.get_value(), relu_graph))
            h_line = always_redraw(lambda: axes.get_h_line(dot.get_left()))
            v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom()))
            self.add(h_line, v_line)
            # self.play(FadeIn(decimal_inp), FadeIn(decimal_out))

            self.play(
                LaggedStart(
                    AnimationGroup(
                        axes_clone.animate.scale(0.10).move_to(l_extras[2][0].get_center()),
                        l_extras[2][1].animate.shift(UP * 0.65).scale(0.8),
                        l_extras[1][1].animate.shift(UP * 0.65).scale(1.0),
                        l_extras[3][1].animate.shift(UP * 0.65).scale(1.0)
                    ),
                    AnimationGroup(
                        FadeIn(decimal_inp),
                        FadeIn(decimal_out)
                    ),
                    lag_ratio=0.5
                )
            )
            self.wait(2)

            table_relu = TableReLU(scale_val=0.65).shift(2.7*LEFT + 0.2*DOWN)
            self.play(FadeIn(table_relu))
            self.wait(2)

            pts_of_interest = [4.0, 2.0, 0.0, -2.0, -4.0]
            curr_pt = 0
            def updater(obj):
                nonlocal curr_pt
                vall = x_tracker.get_value()
                if curr_pt < 5 and vall <= pts_of_interest[curr_pt]:
                    x_pt = pts_of_interest[curr_pt]
                    y_pt = relu_graph.underlying_function(x_pt)
                    curr_pt = curr_pt + 1
                    txt1 = Tex(str(int(x_pt))).scale(0.8).move_to(decimal_inp)
                    txt2 = Tex(str(int(y_pt))).scale(0.8).move_to(decimal_out)
                    anims = table_relu.get_next_segment_animation([txt1, txt2])
                    self.play(*anims)
                obj.set_value(vall)

            decimal_inp.add_updater(updater)
            decimal_out.add_updater(lambda obj: obj.set_value(relu_graph.underlying_function(x_tracker.get_value())))

            self.play(x_tracker.animate.set_value(0), run_time=2)
            self.play(x_tracker.animate.set_value(-4), run_time=2)
            self.play(x_tracker.animate.set_value(4), run_time=4)

            decimal_inp.clear_updaters()
            decimal_out.clear_updaters()

            # self.network_mob = neuron
            self.axes1 = axes
            self.relu_graph1 = relu_graph
            self.decimal_inp = decimal_inp
            self.decimal_out = decimal_out
            self.h_line = h_line
            self.v_line = v_line
            self.dot1 = dot
            self.relu_equation = relu_equation
            self.all_extras = VGroup(*l_extras, txt1_2)
        else:
            self.remove(self.network_mob, self.axes1, self.relu_equation, self.relu_graph1, self.dot1)
            self.remove(self.decimal_inp, self.decimal_out, self.h_line, self.v_line, self.all_extras)
            pass

        return

    def feed_forward(self, input_vector, false_confidence=False, added_anims=None):
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        for i, activation in enumerate(activations):
            self.show_activation_of_layer(i, activation, added_anims)
            added_anims = []

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims=None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        anims = [Transform(layer, active_layer)]
        if layer_index > 0:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index - 1
            )
        anims += added_anims
        self.play(*anims)

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
            self.wait()
        else:
            self.remove(self.vgrp_relu_variants)
            self.sub("")
        return

    def sc_relu_visualize1(self, add=True):
        # To solve this, we will start with a classic problem of teaching XOR function to Neural Network
        screen_grid = ScreenGrid(width=2, height=2, rows=4, columns=4, show_numbers=False, grid_opacity=0.4) \
            # .to_edge(BOTTOM + RIGHT,buff=0.1)
        self.add(screen_grid)

        pos_table = 5*LEFT
        table = TableXOR()
        self.sub("We will take the classic example of learning the XOR using neural networks.")
        self.play(Write(table.rects.shift(pos_table)))
        self.play(
            FadeIn(table.signs_xor0.shift(pos_table), run_time=0.3),
            FadeIn(table.enclosure_xor0.shift(pos_table), run_time=0.3)
        )
        self.play(
            FadeIn(table.signs_xor1.shift(pos_table), run_time=0.3),
            FadeIn(table.enclosure_xor1.shift(pos_table), run_time=0.3)
        )

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
        self.sub("The same is depicted on the graph.")
        self.sub("When x and y are the same, the output is zero (depicted by a x)."
                 "Whereas, when one is 0, and the other is 1, we get 1 as output. (depicted by a blue dot)")
        self.sub("No matter what we do, we cannot pass a line through this graph to segment out the cross from the dot")

        l_values = [
            (PI, [0.5, 0.5, 0]),
            (3*PI/4, [0.5, 0.5, 0]),
            (PI/2, [0.5, 0.5, 0]),
            (PI/4, [0.5, 0.5, 0]),
            (0, [0.5, 0.5, 0]),
            (0, [0.5, 0, 0]),
            (0, [0.5, 1, 0]),
        ]

        ln = DashedLine(0.8*LEFT, 0.8*RIGHT).set_angle(l_values[0][0]).move_to(axes.c2p(*l_values[0][1]))
        ln.save_state()

        def update_pos(obj, alpha):
            print("i= {}".format(i))
            ln.restore()
            diff_angle = l_values[i][0] - l_values[i-1][0]
            diff_center = np.array(l_values[i][1]) - np.array(l_values[i-1][1])
            ln.rotate(alpha*diff_angle)
            ln.shift(alpha*diff_center)
            if alpha == 1:
                ln.save_state()

        self.add(ln)
        for i, [slope, center] in enumerate(l_values):
            if i == 0:
                continue
            # self.play(ln.animate.set_angle(slope).move_to(axes.c2p(*center)))
            # ln.set_angle(slope)
            # ln.move_to(axes.c2p(*center))
            self.play(UpdateFromAlphaFunc(ln, update_pos, run_time=0.7))
            self.wait(1)

        self.wait(2)
        self.play(FadeOut(ln))
        self.remove(ln)

        # # show a line with different orientations...
        # # No amount of squishing of the axes will again solve it for us, since linear transformation leave lines as lines.
        # # And the line connecting the cross, and dots criss-cross with one another.
        #
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
        #
        # # self.remove(screen_grid)
        # self.wait(2)
        #
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

        def transform_fn(pt):
            pt2 = np.matmul(target_mat, pt)
            return pt2

        for i, target_mat in enumerate(l_targets):
            if i in [0, 2, 4]:
                continue
            self.play(axes.animate.apply_function(transform_fn), screen_grid.animate.apply_function(transform_fn), run_time=4)
            self.wait(1)
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
        return

    def sub(self, txt):
        return
        subText = Text(txt, font=FONT).scale(0.6).to_corner(BOTTOM, buff=0.01)
        if self.subText:
            self.play(ReplacementTransform(self.subText, subText, run_time=0.01))
        else:
            self.play(Write(subText, run_time=0.01))
        self.subText = subText
