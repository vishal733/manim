from pyrr import Matrix44

import moderngl_window
from moderngl_window import geometry

from base import CameraWindow


class GeometryBbox(CameraWindow):
    title = "BBox Geometry"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.prog = self.load_program('scene_default/bbox.glsl')
        self.bbox = geometry.bbox()

        self.prog['color'].value = (1, 1, 1)
        self.prog['bb_min'].value = (-2, -2, -2)
        self.prog['bb_max'].value = (2, 2, 2)
        self.prog['m_model'].write(Matrix44.from_translation([0.0, 0.0, -8.0], dtype='f4'))

    def render(self, time: float, frame_time: float):
        self.ctx.clear()

        self.prog['m_proj'].write(self.camera.projection.matrix)
        self.prog['m_cam'].write(self.camera.matrix)
        self.bbox.render(self.prog)


if __name__ == '__main__':
    moderngl_window.run_window_config(GeometryBbox)
