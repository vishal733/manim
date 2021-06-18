from manimlib import *


class OrderMobjects(Scene):
    def construct(self):
        mobs = VGroup(*[
            mob.set_height(4)
            for mob in [Square(color=GREEN, stroke_width=0), Circle(stroke_width=0), Triangle(color=BLUE, stroke_width=0)]
        ])
        mobs.set_fill(opacity=1)
        for i in range(1, 3):
            mobs[i].next_to(mobs[i-1].get_left(), RIGHT, buff=1)
        mobs.move_to(ORIGIN)

        self.add(*mobs)
        self.wait(2)

        print(self.mobjects)

        for mob in self.mobjects:
            name = mob.__class__.__name__
            print(f"Remove {name}")
            self.remove(mob)
            self.add(mob)
            self.wait(2)

        self.wait()

        random.shuffle(self.mobjects)
        self.wait()


# dt parameter is calculated as follows:
# dt = 1 / FPS   ==== dt_calculate = 1 / self.camera.frame_rate
# => Implies that dt varies according to the fps.
class AbstractDtScene(Scene):
    def setup(self):
        path = Line(LEFT*6, RIGHT*6)
        measure = VGroup()
        proportion = 1 / 60
        for i in range(61):
            line = Line(DOWN*0.3, UP*0.3, stroke_width=2)
            line.move_to(path.point_from_proportion(proportion*i))
            measure.add(line)
            if i in [15, 30, 60]:
                arrow = Arrow(UP, DOWN)
                arrow.next_to(line, UP, buff=0.1)
                text = Text(f"{i}", font="Arial", stroke_width=0)
                text.set_height(0.5)
                text.next_to(arrow, UP)
                self.add(arrow, text)
        measure.add(path)

        # Measure lines
        self.measure = measure
        self.measure.start = path.point_from_proportion(0)
        self.dot_distance = path.point_from_proportion(1/60) - path.point_from_proportion(0)

        self.dot = Dot(self.measure.start, color=RED)
        self.add(self.measure)


# If we do not use the dt parameter, then the animation will not be constantly updated.
# Probably this is applicable only for manimce, and not for manimgl
class DtExample1Fail(AbstractDtScene):
    def construct(self):
        def update_dot(mob):
            mob.shift(RIGHT * self.dot_distance)

        dot = self.dot
        dot.add_updater(update_dot)
        self.add(dot)
        self.wait(0.5)
        dot.clear_updaters()
        self.wait()


class DtExample1(AbstractDtScene):
    def construct(self):
        def update_dot(mob, dt):
            print("dt: {}".format(dt))
            mob.shift(RIGHT * self.dot_distance)

        dot = self.dot
        dot.add_updater(update_dot)
        self.add(dot)
        self.wait(0.5)
        dot.clear_updaters()
        self.wait()
