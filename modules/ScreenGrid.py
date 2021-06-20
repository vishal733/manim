from manimlib import *

DISPLAY_CLASS = "CoordScreen"


# General
# self.add, self.remove, self.wait(1), self.play(<animation>, run_time=3)

# absolute positioning -- to_edge, to_corner
# releative positioning -- move_to, next_to, shift
# .rotate(PI/4)

# Text --
# Write, GrowFromCenter, GrowFromEdge(text, LEFT), SpinInFromNothing,
# FadeIn, FadeOut, FadeInFromPoint(text, np.array([-5, 0, 0]))

class Grid(VMobject):
    CONFIG = {
        "height": 6.0,
        "width": 6.0,
    }

    def __init__(self, rows, columns, **kwargs):
        digest_config(self, kwargs, locals())
        VMobject.__init__(self, **kwargs)

    def generate_points(self):
        x_step = self.width / self.columns
        y_step = self.height / self.rows

        for x in np.arange(0, self.width + x_step, x_step):
            self.add(Line(
                [x - self.width / 2., -self.height / 2., 0],
                [x - self.width / 2., self.height / 2., 0],
            ).set_stroke(width=0.5))
        for y in np.arange(0, self.height + y_step, y_step):
            self.add(Line(
                [-self.width / 2., y - self.height / 2., 0],
                [self.width / 2., y - self.height / 2., 0]
            ).set_stroke(width=0.5))


class ScreenGrid(VGroup):
    CONFIG = {
        "rows": 8,
        "columns": 14,
        "height": FRAME_Y_RADIUS * 2,
        "width": 14,
        "grid_stroke": 0.5,
        "grid_color": WHITE,
        "axis_color": RED,
        "axis_stroke": 2,
        "show_points": False,
        "point_radius": 0,
        "labels_scale": 0.5,
        "labels_buff": 0,
        "number_decimals": 2
    }

    def __init__(self, **kwargs):
        VGroup.__init__(self, **kwargs)
        rows = self.rows
        columns = self.columns
        grilla = Grid(width=self.width, height=self.height, rows=rows, columns=columns).set_stroke(self.grid_color,
                                                                                                   self.grid_stroke)

        grilla.generate_points()
        vector_ii = ORIGIN + np.array((-self.width / 2, -self.height / 2, 0))
        vector_id = ORIGIN + np.array((self.width / 2, -self.height / 2, 0))
        vector_si = ORIGIN + np.array((-self.width / 2, self.height / 2, 0))
        vector_sd = ORIGIN + np.array((self.width / 2, self.height / 2, 0))

        ejes_x = Line(LEFT * self.width / 2, RIGHT * self.width / 2)
        ejes_y = Line(DOWN * self.height / 2, UP * self.height / 2)

        ejes = VGroup(ejes_x, ejes_y).set_stroke(self.axis_color, self.axis_stroke)

        divisiones_x = self.width / columns
        divisiones_y = self.height / rows

        direcciones_buff_x = [UP, DOWN]
        direcciones_buff_y = [RIGHT, LEFT]
        dd_buff = [direcciones_buff_x, direcciones_buff_y]
        vectores_inicio_x = [vector_ii, vector_si]
        vectores_inicio_y = [vector_si, vector_sd]
        vectores_inicio = [vectores_inicio_x, vectores_inicio_y]
        tam_buff = [0, 0]
        divisiones = [divisiones_x, divisiones_y]
        orientaciones = [RIGHT, DOWN]
        puntos = VGroup()
        leyendas = VGroup()

        for tipo, division, orientacion, coordenada, vi_c, d_buff in zip([columns, rows], divisiones, orientaciones,
                                                                         [0, 1], vectores_inicio, dd_buff):
            for i in range(1, tipo):
                for v_i, direcciones_buff in zip(vi_c, d_buff):
                    ubicacion = v_i + orientacion * division * i
                    punto = Dot(ubicacion, radius=self.point_radius)
                    coord = round(punto.get_center()[coordenada], self.number_decimals)
                    leyenda = Text("%s" % coord).scale(self.labels_scale)
                    leyenda.set_stroke(width=1)
                    leyenda.next_to(punto, direcciones_buff, buff=self.labels_buff)
                    puntos.add(punto)
                    leyendas.add(leyenda)

        self.add(grilla, ejes, leyendas)
        if self.show_points == True:
            self.add(puntos)


# See https://github.com/3b1b/videos for many, many more

class CoordScreen(Scene):
    def construct(self):
        screen_grid = ScreenGrid()
        self.add(screen_grid)
        obj = Dot()
        obj.move_to(UP*2+RIGHT)
        self.add(obj)
        self.wait()

        reftext = TexText("Text")
        reftext.move_to(3*LEFT+2*UP)
        self.add(reftext)
        obj.move_to(reftext)
        self.wait(2)
        obj.move_to(reftext.get_center() - UP)      # move_to takes geometric center of object into account
        self.wait(2)
        obj.next_to(reftext, RIGHT, buff=5)

        self.wait(2)
        obj.shift(RIGHT)
        obj.shift(RIGHT)


if __name__ == '__main__':
    # sys.argv.append('/home/vishal/.virtualenvs/mnm38/bin/manimgl')
    sys.argv.append('/home/vishal/git/manim/lib/ScreenGrid.py')
    sys.argv.append(DISPLAY_CLASS)
    main()
