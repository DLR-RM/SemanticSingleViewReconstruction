from typing import List

import tensorflow as tf

from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface


class PadLayer(LayerInterface):

    def __init__(self, padding: List[int], name: str):
        super(PadLayer, self).__init__(name=name)
        self.padding = padding

    def call(self, input_tensor, **kwargs):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(input_tensor, [[0, 0], [w_pad, w_pad], [h_pad, h_pad], [d_pad, d_pad], [0, 0]], "REFLECT")
