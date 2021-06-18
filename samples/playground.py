from manimlib import *


class Playground(Scene):
    def construct(self):
        a = Axes(width=6)
        b = Axes(width=6)
        axes = VGroup(a, b).arrange(buff=1)
        axes.center()
        self.add(axes)


class Dragon(Scene):
    def construct(self):
        ax = Axes()
        self.add(ax)
        nbp = NumberPlane()
        self.add(nbp)
        nbp.unfix_from_frame()
        startPoint = np.array((0, -1, 0))
        endPoint = np.array((0, 1, 0))
        obj = Line(start=startPoint, end=endPoint, stroke_width=5, stroke_color=ORANGE)
        obj = VGroup(obj)
        self.play(ShowCreation(obj))
        # self.add(obj)

        new_obj = obj.copy()
        self.play(Rotate(new_obj, angle=PI / 2), about_point=endPoint)
        obj = VGroup(obj, new_obj)
        difff = startPoint - endPoint
        endPoint = endPoint + np.array([-difff[1], difff[0], 0])
        obj.center()

        new_obj = obj.copy()
        self.play(Rotate(new_obj, angle=PI / 2), about_point=endPoint)
        obj = VGroup(obj, new_obj)
        difff = startPoint - endPoint
        endPoint = endPoint + np.array([-difff[1], difff[0], 0])
        # obj.center()

        new_obj = obj.copy()
        new_obj.rotate(angle=PI / 2, about_point=endPoint)
        self.play(Rotate(new_obj, angle=PI / 2), about_point=endPoint)
        obj = VGroup(obj, new_obj)
        difff = startPoint - endPoint
        endPoint = endPoint + np.array([-difff[1], difff[0], 0])
        # obj.center()

        return
