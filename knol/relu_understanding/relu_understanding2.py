from manimlib import *
from knol.relu_understanding.components import SceneComponents


class Scene1(SceneComponents):
    def construct(self):
        self.subText = self.header1 = self.headerline1 = None

        mode = 0
        self.sc_add_heading(mode=mode)
        self.sub("ReLU: Short for Rectified Linear Unit, is one of the most-common activation functions"
                 " across neural networks.")
        self.sc_relu_full_form(mode=mode)

        return


class Scene2(SceneComponents):
    def construct(self):
        self.subText = self.header1 = self.headerline1 = None

        mode = 1
        self.sc_add_heading(mode=mode)
        self.sc_relu_full_form(mode=mode)

        self.sc_neuron_scene()
        # self.sc_neuron_scene(add=False)

        return


class Scene3(SceneComponents):
    def construct(self):
        self.subText = self.header1 = self.headerline1 = None

        mode = 1
        self.sc_add_heading(mode=mode)
        self.sc_relu_full_form(mode=mode)

        self.sc_neuron_scene()
        self.sc_neuron_scene(add=False)
        # self.sc_relu_variants()
        # # self.wait(2)
        # self.sc_relu_variants(add=False)
        # self.wait(2)
        # self.embed()
        # self.sc_relu_visualize1()

        # self.sc_relu_solution()
        return