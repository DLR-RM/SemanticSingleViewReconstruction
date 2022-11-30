from typing import Optional, List
import logging

import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D

from scene_reconstruction.model.layers.layer_interface import LayerInterface
from scene_reconstruction.model.layers.inception_layer import InceptionLayer
from scene_reconstruction.model.layers.resnet_block_layer import ResNetBlockLayer


class NoTreeLayer(LayerInterface):

    def __init__(self, name: str):
        super(NoTreeLayer, self).__init__(name=name)
        self.height: int = LayerInterface.settings("TreeModel/height")
        self._amount_of_filters = LayerInterface.settings("TreeModel/filters_for_level")
        self._amount_of_res_blocks = LayerInterface.settings("TreeModel/amount_of_res_block_per_level")
        self._input_structure = LayerInterface.settings("TreeModel/input_structure")
        self._result_resolution = LayerInterface.settings("TreeModel/result_resolution")
        self._amount_of_filters_in_first_3d_layer = LayerInterface.settings("TreeModel/amount_of_filters_in_first_3D_layer")
        self._desired_amount_of_filters_in_last_2d_layer = self._amount_of_filters_in_first_3d_layer * self._result_resolution
        self._reduce_levels = []
        last_amount_of_filters = 0
        for filter_amount in self._input_structure:
            if filter_amount != -1:
                self._reduce_levels.append(InceptionLayer(filter_amount, name="ReduceLevel"))
                last_amount_of_filters = filter_amount
            else:
                self._reduce_levels.append(MaxPooling2D())

        self._inner_layers = []
        for level_id in range(self.height):
            current_level_amount_of_filters = self._amount_of_filters[level_id]
            current_amount_of_res_blocks = self._amount_of_res_blocks[level_id]
            if current_level_amount_of_filters != last_amount_of_filters:
                # change the amount of filters
                self._inner_layers.append(InceptionLayer(current_level_amount_of_filters, filter_size=1, name=f"Pre_InceptionLayer_{level_id}_before"))
            next_block = ResNetBlockLayer(current_level_amount_of_filters, current_amount_of_res_blocks, name=name + f"_res_net_block_{level_id}")
            self._inner_layers.append(next_block)
            last_amount_of_filters = current_level_amount_of_filters
        if last_amount_of_filters != self._desired_amount_of_filters_in_last_2d_layer:
            # change the amount of filters
            self._inner_layers.append(InceptionLayer(self._desired_amount_of_filters_in_last_2d_layer, filter_size=3, name=f"Pre_InceptionLayer_{level_id}_before"))
        self._final_reshape = tf.keras.layers.Reshape((self._result_resolution, self._result_resolution, self._result_resolution, self._amount_of_filters_in_first_3d_layer))

    @tf.function
    def call(self, input_tensor, training=False):
        x = input_tensor
        for reduce_level in self._reduce_levels:
            x = reduce_level(x)
        for inner_layer in self._inner_layers:
            x = inner_layer(x)
        x = self._final_reshape(x)
        return x


