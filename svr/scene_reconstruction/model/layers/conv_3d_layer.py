from typing import Optional, Union, List
import math

import tensorflow as tf

from tensorflow.keras.layers import Conv3D

from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface
from svr.scene_reconstruction.model.layers.pad_layer import PadLayer


class Conv3DLayer(LayerInterface):

    def __init__(self, amount_of_filter: int, filter_size: Optional[Union[int, List[int]]] = None, dil_value: int = 1, name: str = ""):
        super(LayerInterface, self).__init__(name=name)
        self.use_reflective_padding_3d = LayerInterface.settings("3d_layers/use_reflective_padding_3D")

        if filter_size is None:
            filter_size = LayerInterface.settings("TreeModel/normal_filter_size")

        kernel_regularizer = None
        if LayerInterface.settings("3d_layers/Inception3DLayer/regularizer_scale") > 1e-13:
            kernel_regularizer = tf.keras.regularizers.L2(LayerInterface.settings("3d_layers/Inception3DLayer/regularizer_scale"))

        padding_type = "same"
        if not isinstance(filter_size, list) and not isinstance(filter_size, tuple):
            filter_size = (filter_size, filter_size, filter_size)
        if self.use_reflective_padding_3d:
            padding_type = "valid"
            paddings = [int(math.floor(filter_val * 0.5) * dil_value) for filter_val in filter_size]
            self.pad_layer = PadLayer(paddings, name="PadLayer")
        filter_size = tuple(filter_size)

        self.layer = Conv3D(filters=amount_of_filter, kernel_size=filter_size,
                            padding=padding_type, dilation_rate=(dil_value, dil_value, dil_value),
                            kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                            name=f"inception_layer_dil_{dil_value}_filters_{amount_of_filter}")

    def call(self, input_tensor, training=False):
        x = input_tensor
        if self.use_reflective_padding_3d:
            x = self.pad_layer(x)
        return self.layer(x)



