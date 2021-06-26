import random
import numpy as np
from manimlib import *

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid_inverse(z):
    # z = 0.998*z + 0.001
    assert(np.max(z) <= 1.0 and np.min(z) >= 0.0)
    z = 0.998*z + 0.001
    return np.log(np.true_divide(
        1.0, (np.true_divide(1.0, z) - 1)
    ))

def ReLU(z):
    result = np.array(z)
    result[result < 0] = 0
    return result

def ReLU_prime(z):
    return (np.array(z) > 0).astype('int')


class Network(object):
    def __init__(self, sizes, non_linearity = "sigmoid"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        if non_linearity == "sigmoid":
            self.non_linearity = sigmoid
            self.d_non_linearity = sigmoid_prime
        elif non_linearity == "ReLU":
            self.non_linearity = ReLU
            self.d_non_linearity = ReLU_prime
        else:
            raise Exception("Invalid non_linearity")

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.non_linearity(np.dot(w, a)+b)
        return a

    def get_activation_of_all_layers(self, input_a, n_layers = None):
        if n_layers is None:
            n_layers = self.num_layers
        activations = [input_a.reshape((input_a.size, 1))]
        # for bias, weight in zip(self.biases[:n_layers], self.weights[:n_layers])[:n_layers]:
        for bias, weight in zip(self.biases[:n_layers], self.weights[:n_layers]):
            last_a = activations[-1]
            new_a = self.non_linearity(np.dot(weight, last_a) + bias)
            new_a = new_a.reshape((new_a.size, 1))
            activations.append(new_a)
        return activations

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.non_linearity(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.d_non_linearity(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.d_non_linearity(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return (output_activations-y)


class BigObject(VGroup):
    def __init__(self, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.cl = Circle()
        self.add(self.cl)

        self.sq = Square(side_length=0.5)
        self.add(self.sq)

    def rotate_sq(self, angle):
        self.sq.rotate(angle)


class NetworkScene3(Scene):
    def construct(self):
        self.big_object = BigObject()
        self.add(self.big_object)

        self.big_object.save_state()
        def update(obj, alpha):
            self.big_object.restore()
            obj.shift(RIGHT*alpha)
            obj.rotate_sq(alpha*PI/4)
            # obj.sq.rotate(alpha*PI/4)
            return
        self.play(UpdateFromAlphaFunc(self.big_object, update))
        return


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
        "layer_specific_radius": {1: 1}
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
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
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
        for neuron in neurons:
            neuron.focal_point = lambda: None
            neuron.focal_point.radius = neuron.get_height()/2
            neuron.focal_point.center = neuron.get_center()

            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
            neuron.bias = VGroup()
            neuron.input_obj = None
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

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        self.elabel_groups = VGroup()
        self.bias_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            bias_group = VGroup()
            elabel_group = VGroup()
            for i, (n1, n2) in enumerate(it.product(l1.neurons, l2.neurons)):
                edge = self.update_edge(i, None, n1, n2)
                # edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                elabel_group.add(edge.label)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
                if not n2.bias:
                    bias = self.update_bias(None, n2)
                    n2.bias = bias
                    bias.neuron = n2
                    bias_group.add(bias)
                    elabel_group.add(bias.label)
            self.edge_groups.add(edge_group)
            self.elabel_groups.add(elabel_group)
            self.bias_groups.add(bias_group)
        self.add_to_back(self.elabel_groups)
        self.add_to_back(self.edge_groups)
        self.add_to_back(self.bias_groups)

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
        obj2 = neuron2 if (neuron2.input_obj == None) else neuron2.input_obj
        pt2 = obj2.get_center()
        diff = pt2 - pt1
        lenn = np.linalg.norm(diff)
        start = pt1 + diff/2 * neuron1.get_height() / lenn
        end = pt2 - diff/2 * obj2.get_height() / lenn

        label = edge.label if edge != None else None
        if label == None: label = Tex(r"w_{" + str(idx) + "}").scale(0.5)

        ln = Line(
            start,
            end,
            buff=0.03,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

        edge = ln if (edge == None) else edge.become(ln)
        addn_shift = UP if edge.get_slope() <= 0 else ORIGIN
        label.move_to(edge.get_center()).shift(UP * 0.1 + LEFT * 0.1 + addn_shift * 0.05)
        edge.label = label

        return edge

    def update_bias(self, bias, neuron):
        pt2 = neuron.get_center()
        pt1 = bias.get_start() if (bias != None) else pt2 + np.array([-1, 2, 0])
        diff = pt2 - pt1
        lenn = np.linalg.norm(diff)
        start = pt1
        end = pt2 - diff/2 * neuron.get_height() / lenn

        label = bias.label if bias != None else None
        if label == None: label = Tex(r"b").scale(0.5)

        ln = Line(
            start,
            end,
            buff=0.03,
            stroke_color=self.bias_color,
            stroke_width=self.edge_stroke_width,
        )
        bias = ln if (bias == None) else bias.become(ln)
        label.move_to(bias.get_center()).shift(DOWN * 0.1 + LEFT * 0.1)
        bias.label = label
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

    def label_inputs(self, l, layer_index=0):
        input_labels = VGroup()
        for n, neuron in enumerate(self.layers[layer_index].neurons):
            label = Tex(f"{l}_"+"{"+f"{n + 1}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            input_labels.add(label)
        self.add(input_labels)

    def add_within_neuron(self, l_idx, n_idx):
        neuron = self.layers[l_idx].neurons[n_idx]      # Circle object
        neuron_center = neuron.get_center()

        # cl = Circle(radius=0.4).move_to(neuron_center)
        sq1 = Rectangle(height=1.0, width=1.0).round_corners(0.48).scale(0.8).shift(neuron_center + LEFT * 0.6)
        neuron.add(sq1)
        neuron.input_obj = sq1

        return

    def scaleAndShiftNeuronsInLayer(self, l_idx, scale_factor=None, shift_factor=None):
        vgrp_neurons = self.layers[l_idx][0]

        for neuron in vgrp_neurons:
            if scale_factor:
                neuron.scale(scale_factor)
            if isinstance(shift_factor, np.ndarray):
                neuron.shift(shift_factor)
            neuron.focal_point.center = neuron.get_center()
            neuron.focal_point.radius = neuron.get_height()/2


class NetworkScene(Scene):
    CONFIG = {
        "layer_sizes": [3, 1],
        "network_mob_config": {"layer_specific_radius": {1: 1}},
    }

    # def new_frame_notifier(self):
    #     return

    def construct(self):
        self.network_mob = None

        (network, network_mob) = self.add_network(None, self.network_mob_config)
        self.network = network
        self.network_mob = network_mob
        self.add(self.network_mob)

        # self.remove_random_edges()
        self.network_mob.label_inputs("x", 0)
        # self.network_mob.label_inputs("y", 1)

        # self.network_mob.add_within_neuron(1, 0)
        # self.wait(5)
        self.network_mob.update_edges()

        # return
        self.feed_forward(np.array([[0, 0, 0]]))

        self.network_mob.save_state()

        def modify(obj, alpha):
            obj.restore()
            obj.shift(alpha*LEFT)
            obj.scaleAndShiftNeuronsInLayer(1, scale_factor=1+alpha*3, shift_factor=alpha*RIGHT)
            obj.update_edges()
            return obj

        self.play(UpdateFromAlphaFunc(self.network_mob, modify))
        self.network_mob.update_edges()

        self.network_mob.add_within_neuron(1, 0)

    def add_network(self, network=None, network_mob_config={}):
        if not network:
            network = Network(sizes=self.layer_sizes)
        network_mob = NetworkMobject(
            network,
            **network_mob_config
        )
        return network, network_mob

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

    def remove_random_edges(self, prop=0.9):
        for edge_group in self.network_mob.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)
