from typing import Optional, List, Union

import numpy as np
import tensorflow as tf

from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface
from svr.scene_reconstruction.model.layers.conv_3d_layer import Conv3DLayer


class Inception3DLayer(LayerInterface):

    def __init__(self, amount_of_filters: int, filter_size: Optional[Union[int, List[int]]] = None,
                 used_dil_values: Optional[List[int]] = None, used_dil_ratios: Optional[List[float]] = None,
                 name: str = ""):
        super(Inception3DLayer, self).__init__(name=name)

        self._layers = []
        if filter_size is None:
            filter_size = LayerInterface.settings("TreeModel/normal_filter_size")
        if amount_of_filters == 0:
            raise Exception("The amount of filters can not be zero!")

        if used_dil_values is None:
            # to make sure only a certain amount of filters is used
            used_dil_values = LayerInterface.settings("3d_layers/Inception3DLayer/used_dilation_values")[:amount_of_filters]
        if used_dil_ratios is None:
            # to make sure only a certain amount of filters is used
            used_dil_ratios = LayerInterface.settings("3d_layers/Inception3DLayer/used_dilation_ratios")[:amount_of_filters]
        if np.abs(np.sum(used_dil_ratios) - 1) > 1e-5:
            raise Exception(f"The sum of all dilation ratios has to sum up to one: {used_dil_ratios} is: {np.sum(used_dil_ratios)}")
        if len(used_dil_ratios) != len(used_dil_values):
            raise Exception(f"The amount of elements in the used dilation values and dilation ratios have to be the "
                            f"same: {len(used_dil_values)}, {len(used_dil_ratios)}")

        for dil_value, dil_ratio in zip(used_dil_values, used_dil_ratios):
            amount_of_filters_for_the_current_layer = int(amount_of_filters * dil_ratio)
            self._layers.append(Conv3DLayer(amount_of_filters_for_the_current_layer, filter_size, dil_value))
        self._concat_layers = tf.keras.layers.Concatenate(axis=-1, name="Concat")

    def call(self, input_tensor, training=False):
        outputs = [layer(input_tensor) for layer in self._layers]
        ret = self._concat_layers(outputs)
        return ret
