from manimlib import *
from manimlib.scene.vector_space_scene import LinearTransformationScene
from manimlib.once_useful_constructs.graph_scene import GraphScene


# from manim import *

class VectorRotation(Scene):
    def construct(self):
        dot = Dot(ORIGIN)
        arrow = Arrow(ORIGIN, [2, 1, 0], buff=0)
        vector1 = Vector([3,1])
        numberplane = NumberPlane(faded_line_ratio=2)
        self.add(numberplane, dot, vector1)
        self.wait()
        self.play(Rotate(vector1, angle=7*PI/4, about_point=ORIGIN))
        self.play(Rotate(vector1, angle=-7*PI/4, about_point=ORIGIN))
        self.play(Rotate(vector1, angle=2*PI, about_point=ORIGIN))
        self.play(Rotate(vector1, angle=-2*PI, about_point=ORIGIN))
        self.play(Rotate(vector1, angle=9*PI/4, about_point=ORIGIN))
        self.play(Rotate(vector1, angle=-9*PI/4, about_point=ORIGIN))
        self.wait()


class TransformLinePlot(LinearTransformationScene):
    # def wait(*args, **kwargs):
    #     pass

    def construct(self):
        self.setup()

        d0 = Dot([0, 0, 0])
        d1 = Dot([1, 1, 0])
        d2 = Dot([3, 2, 0])

        v1 = Arrow(d0.get_center(), d1.get_center(), buff=0)
        v2 = Arrow(d0.get_center(), d2.get_center(), buff=0)
        v3 = Arrow(d1.get_center(), d2.get_center(), buff=0)

        self.add_vector(v1)
        self.add_vector(v2)
        self.add_vector(v3)

        self.add_moving_mobject(d1)
        self.add_moving_mobject(d2)

        self.apply_matrix(np.array([[1, -.2], [-.2, 1]]))
        # self.apply_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.wait()


# from manimlib.imports import *

class Shapes(Scene):
    def construct(self):
        ######Code######
        # Making shapes
        circle = Circle()
        square = Square()
        triangle = Polygon(np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([1, -1, 0]))

        # Showing shapes
        self.play(ShowCreation(circle))
        self.play(FadeOut(circle))
        self.play(GrowFromCenter(square))
        self.play(Transform(square, triangle))


from math import cos, sin, pi


class Shapes2(Scene):
    def construct(self):
        #######Code#######
        # Making Shapes
        circle = Circle(color=YELLOW)
        square = Square(color=BLUE_E)
        square.surround(circle)

        rectangle = Rectangle(height=2, width=3, color=RED)
        ring = Annulus(inner_radius=.2, outer_radius=1, color=BLUE)
        ring2 = Annulus(inner_radius=0.6, outer_radius=1, color=BLUE)
        ring3 = Annulus(inner_radius=.2, outer_radius=1, color=BLUE)
        ellipse = Ellipse(width=5, height=3, color=BLUE_E)

        pointers = []
        for i in range(8):
            pointers.append(
                Line(ORIGIN, np.array([cos(pi / 180 * 360 / 8 * i), sin(pi / 180 * 360 / 8 * i), 0]), color=YELLOW))

        # Showing animation
        self.add(circle)
        self.play(FadeIn(square))
        self.play(Transform(square, rectangle))
        self.play(FadeOut(circle), FadeIn(ring))
        self.play(Transform(ring, ring2))
        self.play(Transform(ring2, ring))
        self.play(FadeOut(square), GrowFromCenter(ellipse), Transform(ring2, ring3))
        self.add(*pointers)
        self.wait(2)


class MakeText(Scene):
    def construct(self):
        #######Code#######
        # Making text
        first_line = TexText("Manim is fun")
        second_line = TexText("and useful")
        final_line = TexText("Hope you like it too!", color=BLUE)
        color_final_line = TexText("Hope you like it too!")

        # Coloring
        color_final_line.set_color_by_gradient(BLUE, PURPLE)

        # Position text
        second_line.next_to(first_line, DOWN)

        # Showing text
        self.wait(1)
        self.play(Write(first_line), Write(second_line))
        self.wait(1)
        self.play(FadeOut(second_line), ReplacementTransform(first_line, final_line))
        self.wait(1)
        self.play(Transform(final_line, color_final_line))
        self.wait(2)


class Equations(Scene):
    def construct(self):
        # Making equations
        first_eq = TexText(
            "$$J(\\theta) = -\\frac{1}{m} [\\sum_{i=1}^{m} y^{(i)} \\log{h_{\\theta}(x^{(i)})} + (1-y^{(i)}) \\log{(1-h_{\\theta}(x^{(i)}))}] $$")
        second_eq = ["$J(\\theta_{0}, \\theta_{1})$", "=", "$\\frac{1}{2m}$", "$\\sum\\limits_{i=1}^m$", "(",
                     "$h_{\\theta}(x^{(i)})$", "-", "$y^{(i)}$", "$)^2$"]

        # second_mob = TexText(*second_eq)
        second_mob = [TexText(e) for e in second_eq]

        for i, item in enumerate(second_mob):
            if (i != 0):
                item.next_to(second_mob[i - 1], RIGHT)

        eq2 = VGroup(*second_mob)

        des1 = TexText("With manim, you can write complex equations like this...")
        des2 = TexText("Or this...")
        des3 = TexText("And it looks nice!!")

        # Coloring equations
        eq2.set_color_by_gradient("#33ccff", "#ff00ff")

        # Positioning equations
        des1.shift(2 * UP)
        des2.shift(2 * UP)

        # Animating equations
        self.play(Write(des1))
        self.play(Write(first_eq))
        # self.play(ReplacementTransform(des1, des2), Transform(first_eq, eq2))
        self.play(ReplacementTransform(des1, des2), ReplacementTransform(first_eq, eq2))
        self.wait(1)

        for i, item in enumerate(eq2):
            if (i < 2):
                eq2[i].set_color(color=PURPLE)
            else:
                eq2[i].set_color(color="#00FFFF")

        self.add(eq2)
        self.wait(1)
        # # self.play(FadeOutAndShiftDown(eq2), FadeOutAndShiftDown(first_eq), Transform(des2, des3))
        # # self.play(FadeOut(eq2), FadeOut(first_eq), Transform(des2, des3))
        self.play(FadeOutToPoint(eq2, eq2.get_center() + DOWN),
                  FadeOutToPoint(first_eq, first_eq.get_center() + DOWN), Transform(des2, des3))
        # FadeOutToPoint(eq2, eq2.get_center()+DOWN)
        self.wait(2)


class Graphing(GraphScene):
    CONFIG = {
        "x_min": -5,
        "x_max": 5,
        "y_min": -4,
        "y_max": 4,
        "graph_origin": ORIGIN,
        "function_color": WHITE,
        "axes_color": BLUE
    }

    def construct(self):
        # Make graph
        self.setup_axes(animate=True)
        func_graph = self.get_graph(self.func_to_graph, self.function_color)
        graph_lab = self.get_graph_label(func_graph, label="x^{2}")

        func_graph_2 = self.get_graph(self.func_to_graph_2, self.function_color)
        graph_lab_2 = self.get_graph_label(func_graph_2, label="x^{3}")

        vert_line = self.get_vertical_line_to_graph(1, func_graph, color=YELLOW)

        x = self.coords_to_point(1, self.func_to_graph(1))
        y = self.coords_to_point(0, self.func_to_graph(1))
        horz_line = Line(x, y, color=YELLOW)

        point = Dot(self.coords_to_point(1, self.func_to_graph(1)))

        # Display graph
        self.play(ShowCreation(func_graph), Write(graph_lab))
        self.wait(1)
        self.play(ShowCreation(vert_line))
        self.play(ShowCreation(horz_line))
        self.add(point)
        self.wait(1)
        self.play(Transform(func_graph, func_graph_2), Transform(graph_lab, graph_lab_2))
        self.wait(2)

    def func_to_graph(self, x):
        return (x ** 2)

    def func_to_graph_2(self, x):
        return (x ** 3)


# Three
class ThreeDObjects(ThreeDScene):
    def construct(self):
        sphere = self.get_sphere()
        cube = Cube()
        prism = Prism()
        self.play(ShowCreation(sphere))
        self.play(ReplacementTransform(sphere, cube))
        self.play(ReplacementTransform(cube, prism))
        self.wait(2)


class ThreeDSurface(ParametricSurface):

    def __init__(self, **kwargs):
        # kwargs = {
        # "u_min": -2,
        # "u_max": 2,
        # "v_min": -2,
        # "v_max": 2,
        # "checkerboard_colors": [BLUE_D]
        # }
        kwargs = {
            "u_range": [-2, 2],
            "v_range": [-2, 2],
            "checkerboard_colors": [BLUE_D],
            "color": BLUE_D,
            "prefered_creation_axis": 1,
            "resolution": (101, 101),
            "opacity": 0.5,
        }
        ParametricSurface.__init__(self, self.func, **kwargs)

    def func(self, x, y):
        return np.array([x, y, x ** 2 - y ** 2])
        # return np.array([x, y, x ** 2 + y ** 2])


class Test(ThreeDScene):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "camera_config": {
            # "focal_distance": 0.5,
            # "light_source_position": [1, 1, -2],
            "frame_config": {
                # "frame_shape": (4, 3),
                "center_point": np.array((0., 0., 0.)),
            }
        }
    }

    def construct(self):
        self.camera
        # self.set_camera_orientation(0.6, -0.7853981, 86.6)
        frame = self.camera.frame;
        frame.set_euler_angles(
            theta=-0.7853981,
            phi=0.6,
            # gamma=86.6
        )
        print("Camera centre: {}".format(frame.get_center()))
        # sys.exit()

        surface = ThreeDSurface()
        # self.play(ShowCreation(surface))
        self.add(surface)

        d = Dot(np.array([0, 0, 0]), color=YELLOW)
        self.play(ShowCreation(d))

        # frame.center_point = np.array((5., 5., 0.))
        # frame.refresh_rotation_matrix()
        frame.set_euler_angles(
            theta=0,
            phi=0,
            # gamma=86.6
        )

        self.wait()
        self.move_camera(0.8 * np.pi / 2, -0.45 * np.pi)
        self.begin_ambient_camera_rotation()
        self.wait(9)


class Misc3(GraphScene):
    def construct(self):
        # Create Graph
        self.setup_axes()

        # Define the mean value of the Gaussian PDF
        mean_x = ValueTracker(-3)

        # Define the Gaussian PDF
        pdfX = self.get_graph(
            lambda x: x * ((1 / math.sqrt(2 * PI)) * math.exp(-0.5 * (((x - mean_x.get_value())) ** 2))),
            color=BLUE,
            x_min=-8,
            x_max= 8,
            x_tick_frequency=0.1
        )

        # # Define the updater for the Gaussian PDF
        # pdfX.add_updater(
        #     lambda m: m.become(
        #         self.get_graph(
        #             lambda x: x * ((1 / math.sqrt(2 * PI)) * math.exp(-0.5 * (((x - mean_x.get_value())) ** 2))),
        #             color=BLUE,
        #             x_min=-8,
        #             x_max=8
        #         )
        #     )
        # )

        self.add(pdfX)
        # self.wait()
        # self.play(
        #     ApplyMethod(mean_x.increment_value, 5),
        #     run_time=10,
        # )
        # self.wait()


class RedrawTest(Scene):
    def construct(self):
        # Create x and y coordinates for a set of arrows
        n_arrows = 4
        length = 3
        angles = np.linspace(0, PI, n_arrows)
        x, y = length * np.cos(angles), length * np.sin(angles)

        dot = Dot()
        arrows = [Arrow(dot.get_center(), [x[i], y[i], 0], buff=0) for i in range(n_arrows)]
        lines = [always_redraw(lambda i: Arrow(dot.get_center(), [x[i], y[i], 0], buff=0, tip_length=0.0).set_color(BLUE), i) \
                    for i in range(n_arrows)]

        grow_arrows = [GrowArrow(arrows[i], run_time=2) for i in range(n_arrows)]  # List of GrowArrow animations
        transform = [ReplacementTransform(arrows[i], lines[i], run_time=4) for i in
                     range(n_arrows)]  # List of ReplacementTransform animations

        self.play(FadeIn(dot))
        self.play(*grow_arrows)  # Grow an entire set of arrows simultaneously
        self.wait()
        self.play(*transform)  # Transform entire set of arrows into (always redrawn) lines
        self.wait()
        self.play(dot.animate.shift(2 * DOWN), run_time=3)  # Move dot
        self.wait()


class Complex(Scene):
    def construct(self):
        real = Tex(r'Real Numbers').scale(3)
        realsymbol = Tex('\\mathbb{R}').scale(1).move_to(UP*3)
        realsymbolC = Tex('\\mathbb{C}').scale(1).move_to(UP*3)
        self.play(Write(real))
        self.wait()
        self.play(
            Transform(real, realsymbol)
        )
        n_line = NumberLine(
            x_range=[-5, 5, 1],
            unit_size=0.7,
            include_numbers=True,
            color=WHITE,
        )
        self.play(Write(n_line))
        self.wait()
        i_line = NumberLine(
            x_range=[-5, 5, 1],
            unit_size=0.5,
            color=WHITE,
        ).set_angle(PI/2)
        self.play(
            Transform(n_line, i_line.move_to(DOWN*(1/2))),
            ReplacementTransform(real, realsymbolC),
        )
        i0 = Tex(r'0')
        i1 = Tex(r'1i')
        i2 = Tex(r'2i')
        i3 = Tex(r'3i')
        i4 = Tex(r'4i')
        i5 = Tex(r'5i')
        i_1 = Tex(r'-1i')
        i_2 = Tex(r'-2i')
        i_3 = Tex(r'-3i')
        i_4 = Tex(r'-4i')
        i_5 = Tex(r'-5i')
        i0.next_to(i_line, RIGHT).scale(0.6)
        i1.next_to(i0, UP*0.9).scale(0.6)
        i2.next_to(i1, UP*0.9).scale(0.6)
        i3.next_to(i2, UP*0.9).scale(0.6)
        i4.next_to(i3, UP*0.9).scale(0.6)
        i5.next_to(i4, UP*0.9).scale(0.6)
        i_1.next_to(i0, DOWN*0.9).scale(0.6)
        i_2.next_to(i_1, DOWN*0.9).scale(0.6)
        i_3.next_to(i_2, DOWN*0.9).scale(0.6)
        i_4.next_to(i_3, DOWN*0.9).scale(0.6)
        i_5.next_to(i_4, DOWN*0.9).scale(0.6)
        grp_items = VGroup(i0, i1, i2, i3, i4, i5, i_1, i_2, i_3, i_4, i_5)
        self.play(
            FadeIn(grp_items)
        )
        self.wait()


class VField(Scene):

    @staticmethod
    def scale_func(func, scalar):
        return lambda p: func(p * scalar)

    def construct(self):
        npl = NumberPlane(background_line_style={
            "stroke_color": GREEN_D,
            "stroke_width": 1,
            "stroke_opacity": 1,
        })
        self.add(npl)
        vectors = []
        tail = [x * RIGHT + y * UP
                for x in np.arange(-3, 3, 1)
                for y in np.arange(-3, 3, 1)]

        for p in tail:
            head = 0.37 * RIGHT + 0.63 * UP
            vector = Vector(head, thickness=0.03).shift(p)
            vectors.append(vector)
        vector_field = VGroup(*vectors)

        func = lambda pos: np.sin(pos[1]) * RIGHT + np.cos(pos[0]) * UP

        self.play(FadeIn(vector_field))
        self.wait(1)
        func = VField.scale_func(func, 0.5)
        self.play(vector_field.animate.become(VectorField(func), npl))
        # self.play(vector_field.animate.become(ArrowVectorField(func)))


class ArcBetweenPointsExample(Scene):
    def construct(self):
        cl = Circle(radius=2.0).set_stroke(GREY_C)
        dot1 = Dot(np.array([2, 0, 0]), radius=0.06, color=GREEN_C)
        dot2 = Dot(np.array([0, 2, 0]), radius=0.06, color=GREEN_C)
        txt1 = Tex("(2, 0)").scale(0.6).set_color(BLUE_C).next_to(dot1)
        txt2 = Tex("(0, 2)").scale(0.6).set_color(BLUE_C).next_to(dot2, UP)
        self.add(cl, dot1, dot2, txt1, txt2)

        arc1 = ArcBetweenPoints(start=2 * RIGHT, end=2 * UP, stroke_color=YELLOW)
        self.play(ShowCreation(arc1))


class Equations(Scene):
    def construct(self):
        scale = 1.5
        # Make and arrange the equations
        eq1 = Tex("0 - 1 = -1").shift([0, 3, 0]).scale(scale).align_to(LEFT)
        eq_list = [eq1]
        for i in range(4):
            eq = eq1.copy()
            eq_list.append(eq)
        eq6 = Tex("7 - 1 = +6").shift([0, 3, 0]).scale(scale).align_to(eq1, LEFT)
        eq_list.append(eq6)
        for i in range(len(eq_list)):
            eq_list[i].shift(DOWN * 1.0 * (i + 0.5))

        # Write them out together
        eq_g = VGroup(*eq_list)
        self.play(Write(eq_g), run_time=3)

        # Labels
        payout = Text("Payout").shift([-1.7, 3.3, 0]).scale(0.5)
        cost = Text("Cost").scale(0.5).align_to(payout, UP).shift([-0.35, 0, 0])
        net = Text("Net gain").scale(0.5).align_to(payout, UP).shift([1.7, 0, 0])
        labels = VGroup(payout, cost, net)
        self.play(Write(labels), run_time=2)

        self.wait(1)
        # Transform
        eqs_combined = Tex("7 - 6 = +1").scale(scale)
        group_animation = AnimationGroup(
            ReplacementTransform(eq_list[0], eqs_combined),
            ReplacementTransform(eq_list[1], eqs_combined),
            ReplacementTransform(eq_list[2], eqs_combined),
            ReplacementTransform(eq_list[3], eqs_combined),
            ReplacementTransform(eq_list[4], eqs_combined),
            ReplacementTransform(eq_list[5], eqs_combined),
            lag_ratio=.1
        )
        self.play(
            group_animation,
            ApplyMethod(labels.shift, [0, -2.5, 0])
        )
