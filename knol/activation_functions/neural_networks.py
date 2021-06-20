from lib.NeuralNetwork import NeuralNetworkMobject
from manimlib import *


class MyNeuralNetwork(Scene):
    def construct(self):
        myNetwork = NeuralNetworkMobject([10000, 5, 1])
        myNetwork.label_inputs('x')
        myNetwork.label_outputs('\hat{y}')
        myNetwork.label_outputs_text(['isPrediction'])
        myNetwork.label_hidden_layers('h')

        myNetwork.scale(0.75)
        self.play(Write(myNetwork))
        self.wait()
