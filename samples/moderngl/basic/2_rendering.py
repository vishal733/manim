# Reference - https://moderngl.readthedocs.io/en/latest/the_guide/rendering.html
import os, sys
import moderngl
import numpy as np

from PIL import Image

ctx = moderngl.create_standalone_context()

prog = ctx.program(
    vertex_shader='''
        #version 330

        in vec2 in_vert1;
        in vec3 in_color;

        out vec3 v_color;

        void main() {
            v_color = in_color;
            gl_Position = vec4(in_vert1, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330

        in vec3 v_color;

        out vec3 f_color;

        void main() {
            f_color = v_color;
        }
    ''',
)

x = np.linspace(-1.0, 1.0, 50)          # (50, )
y = np.random.rand(50) - 0.5            # (50, )
r = np.ones(50)                         # (50, )
g = np.zeros(50)                        # (50, )
b = np.zeros(50)                        # (50, )

vertices = np.dstack([x, y, r, g, b])   # (1, 50, 5)
# print(y.shape)
# sys.exit()

vbo = ctx.buffer(vertices.astype('f4').tobytes())                   # vbo == vertex buffer object
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert1', 'in_color')     # vao == vertex array object

fbo = ctx.simple_framebuffer((512, 512))
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 1.0)
vao.render(moderngl.LINE_STRIP)

Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
