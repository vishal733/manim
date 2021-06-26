import random
import numpy as np
from manimlib import *
from modules.placeholders import FakeObject
from knol.relu_understanding.network import Network


class Neuron(Circle):
    def __init__(self, *args, **kwargs):
        Circle.__init__(self, *args, **kwargs)
        self.focal_pt = None
        self.edges_in = VGroup()
        self.edges_out = VGroup()
        self.bias = VGroup()
        self.input_obj = None
        self.alpha = 0
        return

    def update_input_obj(self, obj=None):
        self.input_obj = obj
        if self.input_obj == None:
            self.alpha = 0

    def set_transition_alpha(self, alpha):
        if self.input_obj != None:
            self.alpha = alpha
        else:
            self.alpha = 0

    @property
    def target_center(self):
        cl = self
        if self.input_obj:
            cl = self.input_obj
        return cl.get_center()

    @property
    def target_height(self):
        cl = self
        if self.input_obj:
            cl = self.input_obj
        return cl.get_height() / 2

    @property
    def working_center(self):
        cl = self
        out_center = cl.get_center()
        if self.alpha == 1:
            cl = self.input_obj
            out_center = cl.get_center()
        elif self.alpha != 0:
            out_center = cl.get_center()
            diff_center = self.input_obj.get_center() - out_center
            out_center = out_center + self.alpha*diff_center
        return out_center

    @property
    def working_height(self):
        cl = self
        out_height = cl.get_height()
        if self.alpha == 1:
            cl = self.input_obj
            out_height = cl.get_height()
        elif self.alpha != 0:
            out_height = cl.get_height()
            diff_height = self.input_obj.get_height() - out_height
            out_height = out_height + self.alpha*diff_height
        return out_height

    # @focus_center.setter
    # def focus_center(self, value):
    #     print("Setting value...")
    #     if value < -273.15:
    #         raise ValueError("Temperature below -273 is not possible")
    #     self._temperature = value


class NetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.3,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": 1.75*LARGE_BUFF,
        "neuron_stroke_color": BLUE,
        "neuron_stroke_width": 3,
        "neuron_fill_color": GREEN,
        "edge_color": GREY_B,
        "edge_stroke_width": 2,
        "bias_color": GREEN_B,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 4,
        "brace_for_large_layers": False,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "layer_specific_radius": {1: 1},
        "layer_labels": []
    }

    def __init__(self, neural_network, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neural_network = neural_network
        self.layer_sizes = neural_network.sizes
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(i, size)
            for i, size in enumerate(self.layer_sizes)
        ])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()

    def get_layer(self, layer_index, size):
        layer = VGroup()
        layer.neurons = None
        layer.brace = None
        layer.brace_label = None
        layer.brace_vgrp = None

        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Neuron(
                radius=self.neuron_radius if layer_index not in self.layer_specific_radius
                        else self.neuron_radius * self.layer_specific_radius[layer_index],
                stroke_color=self.neuron_stroke_color,
                stroke_width=self.neuron_stroke_width,
                fill_color=self.neuron_fill_color,
                fill_opacity=0,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange(
            DOWN, buff=self.neuron_to_neuron_buff
        )

        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)
        if layer_index == 0:
            brace = Brace(layer, LEFT)
            brace_label = brace.get_tex(str("Inputs")).scale(0.75)
            layer.brace = brace
            layer.brace_label = brace_label
            layer.brace_vgrp = VGroup(brace, brace_label)
            layer.add(brace, brace_label)

        return layer

    def set_opacity_brace_layers(self, layer_index, opacity=0.0):
        layer = self.layers[layer_index]
        if layer.brace_vgrp:
            layer.brace_vgrp.set_opacity(opacity)
        # if layer.brace:
        #     layer.brace.set_opacity(opacity)
        # if layer.brace_label:
        #     layer.brace_label.set_opacity(opacity)

    def add_edges(self):
        self.edge_groups = VGroup()
        self.bias_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            bias_group = VGroup()
            for i, (n1, n2) in enumerate(it.product(l1.neurons, l2.neurons)):
                edge = self.update_edge(i, None, n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
                if not n2.bias:
                    bias = self.update_bias(None, n2)
                    n2.bias = bias
                    bias.neuron = n2
                    bias_group.add(bias)
            self.edge_groups.add(edge_group)
            self.bias_groups.add(bias_group)

        self.add(self.edge_groups)
        self.add(FakeObject(1))
        self.add(self.bias_groups)

    def update_edges(self):
        for j, (l1, l2) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            edge_group = self.edge_groups[j]
            for i, (n1, n2) in enumerate(it.product(l1.neurons, l2.neurons)):
                edge = edge_group[i]
                self.update_edge(i, edge, n1, n2)

        for bias_group in self.bias_groups:
            for bias in bias_group:
                self.update_bias(bias, bias.neuron)

    def update_edge(self, idx, edge, neuron1, neuron2):
        pt1 = neuron1.get_center()
        # obj2 = neuron2 if (neuron2.input_obj == None) else neuron2.input_obj
        # obj2 = neuron2.wor
        pt2 = neuron2.working_center
        diff = pt2 - pt1
        lenn = np.linalg.norm(diff)
        start = pt1 + diff * (neuron1.get_height()/2+0.06) / lenn
        end = pt2 - diff * (neuron2.working_height/2 + 0.06) / lenn

        if edge:
            # IMPORTANT: First remove the label, prior to calling become!
            label = edge.label
            edge.remove(label)
        else:
            label = Tex(r"w_{" + str(idx) + "}").scale(0.5)

        ln = Line(
            start,
            end,
            buff=0.00,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

        edge = ln if (edge == None) else edge.become(ln)
        addn_shift = UP if edge.get_slope() <= 0 else ORIGIN
        label.move_to(edge.get_center()).shift(UP * 0.1 + LEFT * 0.1 + addn_shift * 0.05)
        edge.label = label
        edge.add(label)

        return edge

    def update_bias(self, bias, neuron):
        pt2 = neuron.working_center
        pt1 = bias.get_start() if (bias != None) else pt2 + np.array([-1, 2, 0])
        diff = pt2 - pt1
        lenn = np.linalg.norm(diff)
        start = pt1
        end = pt2 - (diff)/lenn * (neuron.working_height/2 + 0.06)

        if bias:
            label = bias.label
            bias.remove(label)
        else:
            label = Tex(r"b").scale(0.5)

        ln = Line(
            start,
            end,
            buff=0.00,
            stroke_color=self.bias_color,
            stroke_width=self.edge_stroke_width,
        )
        bias = ln if (bias == None) else bias.become(ln)
        label.move_to(bias.get_center()).shift(DOWN * 0.1 + LEFT * 0.1)
        bias.label = label
        bias.add(label)

        return bias

    def get_active_layer(self, layer_index, activation_vector):
        layer = self.layers[layer_index].deepcopy()
        self.activate_layer(layer, activation_vector)
        return layer

    def activate_layer(self, layer, activation_vector):
        n_neurons = len(layer.neurons)
        av = activation_vector

        def arr_to_num(arr):
            return (np.sum(arr > 0.1) / float(len(arr))) ** (1. / 3)

        if len(av) > n_neurons:
            if self.average_shown_activation_of_large_layer:
                indices = np.arange(n_neurons)
                indices *= int(len(av) / n_neurons)
                indices = list(indices)
                indices.append(len(av))
                av = np.array([
                    arr_to_num(av[i1:i2])
                    for i1, i2 in zip(indices[:-1], indices[1:])
                ])
            else:
                av = np.append(
                    av[:n_neurons / 2],
                    av[-n_neurons / 2:],
                )
        for activation, neuron in zip(av, layer.neurons):
            neuron.set_fill(
                color=self.neuron_fill_color,
                opacity=activation
            )
        return layer

    def activate_layers(self, input_vector):
        activations = self.neural_network.get_activation_of_all_layers(input_vector)
        for activation, layer in zip(activations, self.layers):
            self.activate_layer(layer, activation)

    def deactivate_layers(self):
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.layers
        ]))
        all_neurons.set_fill(opacity=0)
        return self

    def get_edge_propogation_animations(self, index):
        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width=1.5 * self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy,
            run_time=self.edge_propogation_time,
            lag_ratio=0.5
        )]

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(str(n))
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width() * RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def set_label_layer(self, l, layer_index=0):
        input_labels = VGroup()
        for n, neuron in enumerate(self.layers[layer_index].neurons):
            label = Tex(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            input_labels.add(label)
            neuron.add(label)

    def add_within_tiny_neuron(self, l_idx, n_idx):
        neuron = self.layers[l_idx].neurons[n_idx]  # Circle object
        neuron_center = neuron.get_center()

        arrow_a = Arrow(neuron.get_right(), neuron.get_right() + RIGHT * 0.5, buff=0)
        txta_1 = Tex(r"a").move_to(arrow_a.get_right()).scale(0.8).shift(RIGHT * 0.3)
        extras1 = VGroup(arrow_a, txta_1)

        return neuron, [extras1]

    @staticmethod
    def get_neuron_internal_contents(center=ORIGIN, shift=ORIGIN):
        arrow_len = 0.5
        sq2 = Rectangle(height=0.8, width=0.8).round_corners(0.1)
        txt2_1 = Tex(r"").scale(1.0).move_to(sq2.get_center())

        arrow_z2a = Arrow(sq2.get_left() - np.array([arrow_len, 0, 0]), sq2.get_left(), buff=0, thickness=0.02)
        txtz2a_1 = Tex(r"").move_to(arrow_z2a.get_left()).scale(0.8).shift(LEFT * 0.3)

        arrow_a = Arrow(sq2.get_right(), sq2.get_right() + np.array([arrow_len, 0, 0]), buff=0, thickness=0.02)
        txta_1 = Tex(r"").move_to(arrow_a.get_right()).scale(0.8).shift(RIGHT * 0.3)

        extras3 = VGroup(arrow_z2a, txtz2a_1).shift(shift)
        extras2 = VGroup(sq2, txt2_1).shift(shift)
        extras4 = VGroup(arrow_a, txta_1).shift(shift)

        l_extras = [extras3, extras2, extras4]

        return l_extras

    def add_within_neuron(self, l_idx, n_idx):
        neuron = self.layers[l_idx].neurons[n_idx]      # Circle object
        neuron_center = neuron.get_center()

        l_extras = []
        container = Circle(radius=0.4, stroke_color=BLUE
                           ).move_to(neuron_center).shift(LEFT * 0.85)
        # contaner = Rectangle(height=1.0, width=1.0).round_corners(0.48).scale(0.8).shift(neuron_center + LEFT * 0.6)
        neuron.input_obj = container
        txt1_1 = Tex(r"\sum").scale(0.5).move_to(container.get_center())
        extras1 = VGroup(container, txt1_1)

        sq2 = Rectangle(height=0.8, width=0.8).shift(neuron_center + RIGHT * 0.70).round_corners(0.1)
        txt2_1 = Tex(r"f").scale(1.0).move_to(sq2.get_center())
        extras2 = VGroup(sq2, txt2_1)

        arrow_z2a = Arrow(container.get_right(), sq2.get_left(), buff=0)
        txtz2a_1 = Tex(r"z").move_to(arrow_z2a.get_center()).scale(0.8).shift(UP * 0.3)
        extras3 = VGroup(arrow_z2a, txtz2a_1)

        arrow_a = Arrow(sq2.get_right(), neuron.get_right()+RIGHT*0.5, buff=0)
        txta_1 = Tex(r"a").move_to(arrow_a.get_right()).scale(0.8).shift(RIGHT * 0.3)
        extras4 = VGroup(arrow_a, txta_1)

        l_extras.append(extras1)
        l_extras.append(extras3)
        l_extras.append(extras2)
        l_extras.append(extras4)

        return neuron, l_extras

    def get_neuron(self, l_idx, n_idx):
        return self.layers[l_idx].neurons[n_idx]

    def scaleAndShiftNeuronsInLayer(self, l_idx, scale_factor=None, shift_factor=None):
        vgrp_neurons = self.layers[l_idx][0]

        for neuron in vgrp_neurons:
            if scale_factor:
                neuron.scale(scale_factor)
            if isinstance(shift_factor, np.ndarray):
                neuron.shift(shift_factor)

    @staticmethod
    def moveInStepsToNewCenter(obj, l_idx, alpha):
        vgrp_neurons = obj.layers[l_idx][0]
        for neuron in vgrp_neurons:
            height, center = neuron.focal_pt.get_height(), neuron.focal_pt.get_center()
            target_height, target_center = neuron.input_obj.get_height(), neuron.input_obj.get_center()
            diff_height = target_height - height
            diff_center = target_center - center
            if diff_center != 0:
                neuron.focal_pt.move_to(center + alpha*diff_center)
            if diff_height != 0:
                neuron.focal_pt.set_height(height + alpha*diff_height)

        return

    @staticmethod
    def add_network(layer_sizes, network=None, network_mob_config={}):
        if not network:
            network = Network(sizes=layer_sizes)
        network_mob = NetworkMobject(
            network,
            **network_mob_config
        )
        return network, network_mob

    def remove_random_edges(self, prop=0.9):
        for edge_group in self.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)

