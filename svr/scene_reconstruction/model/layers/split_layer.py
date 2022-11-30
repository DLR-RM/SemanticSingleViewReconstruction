import tensorflow as tf

from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface
from svr.scene_reconstruction.model.layers.inception_layer import InceptionLayer
from svr.scene_reconstruction.model.layers.resnet_block_layer import ResNetBlockLayer


class SplitLayer(LayerInterface):

    def __init__(self, amount_of_filters: int, amount_of_residual_blocks: int = 0, name=""):
        super(SplitLayer, self).__init__(name=name + "_split_layer")

        if amount_of_residual_blocks == 0:
            self._first_out = InceptionLayer(amount_of_filters, name=name + "_first_split")
            self._second_out = InceptionLayer(amount_of_filters, name=name + "_second_split")
        else:
            self._first_out = ResNetBlockLayer(amount_of_filters, amount_of_residual_blocks, name=name + "_front_split")
            self._second_out = ResNetBlockLayer(amount_of_filters, amount_of_residual_blocks, name=name + "_back_split")

    def call(self, input_tensor, training=False):
        return self._first_out(input_tensor), self._second_out(input_tensor)


