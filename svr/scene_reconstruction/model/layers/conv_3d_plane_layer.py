from typing import Optional

from svr.scene_reconstruction.model.layers.inception_3d_layer import Inception3DLayer
from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface


class Conv3DPlaneLayer(LayerInterface):

    def __init__(self, amount_of_filter: int, filter_size: Optional[int] = None, name: str = ""):
        super(LayerInterface, self).__init__(name=name)
        if filter_size is None:
            filter_size = LayerInterface.settings("TreeModel/normal_filter_size")

        self.use_plane_for_separable_3d = LayerInterface.settings("3d_layers/for_separable_3D/use_plane_for_separable_3D")
        self.dil_values_for_3d_separable = LayerInterface.settings("3d_layers/for_separable_3D/dil_values_for_separable")
        self.dil_ratio_for_separable = LayerInterface.settings("3d_layers/for_separable_3D/dil_ratio_for_separable")
        self._layers = []
        for i in range(3):
            if self.use_plane_for_separable_3d:
                kernel_size = [filter_size, filter_size, filter_size]
                kernel_size[i] = 1
            else:
                kernel_size = [1, 1, 1]
                kernel_size[i] = filter_size
            next_layer = Inception3DLayer(amount_of_filter, kernel_size, self.dil_values_for_3d_separable,
                                          self.dil_ratio_for_separable)
            self._layers.append(next_layer)

    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self._layers:
            x = layer(x)
        return x
