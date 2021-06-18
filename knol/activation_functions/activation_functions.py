import numpy as np

from manimlib import *
import itertools, functools

L_COLORS = itertools.cycle([BLUE, YELLOW, GREEN, PINK, MAROON])
L_COLORS = itertools.cycle([MAROON])

# Could construct the formula part-by-part...

def swish(x, beta=1):
    return (x * sigmoid(beta * x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softplus(x):
    return np.log(1 + np.exp(x))


class Heading(Text):
    def __init__(self, txt, *args, **kwargs):
        MAX_WORD_PER_LINE = 15
        if not kwargs: kwargs = {}
        kwargs['lsh'] = 1.3
        kwargs['size'] = 0.9
        parts = txt.split(" ")
        txt2 = parts[0]
        L = len(txt2)
        for part in parts[1:]:
            if L + len(part) > MAX_WORD_PER_LINE:
                txt2 += "\n"
                L = 0
            else:
                L += len(part)
            txt2 += " " + part

        super().__init__(txt2, *args, **kwargs)


# def y_ceil_wrapper(func, y_range=(-1, 1)):
#     def func2(x, dt=0.01):
#         val = func(x)
#         if (val < y_range[-1] or val > y_range[1]):
#             return None
#     return func2

def y_ceil_wrapper(func, y_range=(-2.5, 4.5)):
    def func2(x, dt=0.01):
        val = func(x)
        if not (y_range[0] < val < y_range[1]):
            return np.NAN
        return val

    return func2


def derivative(func):
    def func2(x, dt=0.01):
        val = (func(x+dt) - func(x-dt))/(2*dt)
        return val
    return func2


class ActivationFunctions(Scene):
    def construct(self):

        density = 4 / 6
        density = 7/12
        density = 1.1*5/8
        density2 = 1.5*density
        axes = Axes((-5, 6), (-2, 5),
                    width=11 * density, height=7 * density,
                    x_axis_config={"include_tip": True},
                    y_axis_config={"include_tip": True}
                    )
        axes.add_coordinate_labels(font_size=16)
        axes.shift(UP * 1.5 + 2.5*RIGHT)

        color_axes2 = GREY_D
        axes2 = Axes((-5, 6), (-1, 1),
                     width=11 * density, height=2 * density2,
                     x_axis_config={"color": color_axes2, "stroke_color": color_axes2},
                     y_axis_config={"color": color_axes2, "stroke_color": color_axes2,
                                    "include_tip": False, "include_ticks": True, "xtra_pad_start": 0.1, "xtra_pad_end": 0.2}
                     )
        # axes2.add_coordinate_labels(font_size=16, color=color_axes2)
        axes2.shift(DOWN * 2.5 + 2.5*RIGHT)

        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        self.play(FadeIn(axes2, lag_ratio=0.01, run_time=1))

        a = 0.1
        alpha = 0.1
        common_params = {"use_smoothing": False}
        l_details = [
            ['step', Heading("Step"),
                Tex(r"""f(x) = \begin{cases}1 & x \geq 0\\0 & x < 0\end{cases}"""),
                lambda x: 1 if x > 0 else 0, {"discontinuities": [0]}, None
            ],
            ['linear', Heading("Linear"), Tex(r"f(x)=x"),
                lambda x: x, {}, None
            ],
            [
                'sigmoid', Heading("Sigmoid"),
                Tex(r"f(x) = \frac{1}{1 + e^{-x}}"),
                lambda x: 1 / (1 + np.exp(-x)), {}, None
            ],
            [
                'tanh', Heading("Tanh"),
                Tex(r"f(x) = \frac{e^{-x} - e^{-x}}{e^{-x} + e^{-x}}"),
                lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), {}, None
            ],
            [
                'relu', Heading("ReLU"),
                Tex(r"""
                f(x) = max(x, 0)
\\
f(x) = \begin{cases}x & x \geq 0\\0 & x < 0\end{cases}
                """),
                lambda x: max(x, 0), {"discontinuities": [0]}, None
            ],
            [
                'leaky-relu', Heading("Leaky ReLU"),
                Tex(r"f(x) = \begin{cases}x & x \geq 0\\0.01 \cdot x & x < 0\end{cases}"),
                lambda x: x if x > 0 else 0.01 * x, {"discontinuities": [0]}, None
            ],
            [
                'parametrized-relu', Heading("Parametrized ReLU"),
                Tex(r"f(x) = \begin{cases}x & x \geq 0\\a \cdot x & x < 0\end{cases}"),
                lambda x: x if x > 0 else a * x, {"discontinuities": [0]}, None
            ],  # a is a learnable parameter
            [
                'exponential-linear-unit', Heading("Exponential Linear Unit"),
                Tex(r"f(x) = \begin{cases}z & z \geq 0\\\alpha \cdot (e^{z}-1) & z < 0\end{cases}"),
                lambda x: x if x > 0 else alpha * (np.exp(x) - 1), {}, None
            ],
            [
                'swish', Heading("Swish"),
                # Tex(r"f(x)=x \cdot sigmoid(x)"),
                Tex(r"f(x)=x \cdot \frac{1}{1 + e^{-x}}"),
                swish, {}, None
            ],  # Swish is not monotonic
            [
                'mish', Heading("Mish"),
                # Tex(r"f(z)=z \cdot tanh(softplus(z))"),
                Tex(r"f(z)=z \cdot tanh(\ln(1+e^{z}))"),
                lambda x: x * tanh(softplus(x)), {}, None
            ],
        ]

        prev_graph = prev_label = prev_deriv = prev_deriv2 = prev_riemann_rectangles = prev_formula = None
        for i, details in enumerate(l_details):
            [_, display_name, formula, func, params, x_range] = details
            if not x_range: x_range = [-5, 6]
            params.update(common_params)
            color = next(L_COLORS)
            graph = axes.get_graph(y_ceil_wrapper(func), color=color, stroke_width=2.0, x_range=x_range, **params)
            derivative_func = functools.partial(axes.slope_of_tangent, graph=graph)
            deriv = axes.get_graph(derivative_func, color=LIGHT_BROWN, stroke_width=2.0, x_range=x_range, **params)
            label = display_name.to_edge(UL).shift(RIGHT)
            formula = formula.to_corner(LEFT)

            graph2 = axes.get_graph(func, color=color, stroke_width=2.0, x_range=x_range, **params)
            derivative_func = functools.partial(axes.slope_of_tangent, graph=graph2)
            deriv2 = axes2.get_graph(derivative_func, color=LIGHT_BROWN, stroke_width=4.0, x_range=x_range, **params)
            # deriv2 = axes2.get_graph(derivative(func), color=LIGHT_BROWN, stroke_width=4.0, x_range=x_range, **params)
            # deriv2 = axes2.get_derivative_graph(graph)

            riemann_rectangles = axes.get_riemann_rectangles(
                graph,
                x_range=x_range,
                dx=max(graph.t_range[2] / 500, 0.005),
                stroke_width=0,
                colors=[color, color],
                custom=True,
                fill_opacity=0.1
            )
            if i == 0:
                self.add(deriv2)
                self.play(ShowCreation(graph), FadeIn(label, RIGHT))
                self.add(formula)
                # self.wait(1)
                self.add(riemann_rectangles)
                self.wait(1)
            else:
                self.remove(prev_deriv, prev_deriv2, prev_riemann_rectangles, prev_formula)

                self.play(ReplacementTransform(prev_graph, graph),
                          FadeTransform(prev_label, label), run_time=0.5)
                # self.wait(1)
                self.add(formula)
                self.add(riemann_rectangles)
                self.play(Write(deriv2))
                self.wait(1)
            # self.remove(formula)
            prev_graph = graph
            prev_label = label
            prev_deriv = deriv
            prev_deriv2 = deriv2
            prev_riemann_rectangles = riemann_rectangles
            prev_formula = formula
