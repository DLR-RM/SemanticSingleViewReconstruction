from typing import Optional, List

import tensorflow as tf

from .layer_interface import LayerInterface
from .inception_layer import InceptionLayer


class ResNetBlockLayer(LayerInterface):

    def __init__(self, amount_of_filters: int, amount_of_residual_blocks: int, name: str = ""):
        super(ResNetBlockLayer, self).__init__(name=name)
        self._layers = []

        for i in range(amount_of_residual_blocks):
            first_conv_layer = InceptionLayer(amount_of_filters, name=f"first_{i}")
            first_act_layer = tf.keras.layers.ReLU(name=f"relu_first_{i}")
            second_conv_layer = InceptionLayer(amount_of_filters, name=f"second_{i}")
            second_act_layer = tf.keras.layers.ReLU(name=f"relu_second_{i}")
            add_layer = tf.keras.layers.Add(name=f"add_{i}")
            self._layers.append((first_conv_layer, first_act_layer, second_conv_layer, second_act_layer, add_layer))

    def call(self, input_tensor, training=False):
        block_input = input_tensor
        x = input_tensor
        for layer in self._layers:
            first_conv_layer, first_act_layer, second_conv_layer, second_act_layer, add_layer = layer
            x = first_conv_layer(x)
            x = first_act_layer(x)
            x = second_conv_layer(x)
            temp = add_layer([x, block_input])
            x = second_act_layer(temp)
            block_input = x

        return x


