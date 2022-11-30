import tensorflow as tf
from tensorflow.keras import Model

from svr.scene_reconstruction.model.layers.conv_3d_layer import Conv3DLayer
from svr.scene_reconstruction.model.layers.conv_3d_plane_layer import Conv3DPlaneLayer
from svr.scene_reconstruction.model.layers.inception_3d_layer import Inception3DLayer
from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface
from svr.scene_reconstruction.model.layers.tree_layer import TreeLayer
from svr.scene_reconstruction.utility.settings_manager import SettingsManager


class TreeModel(Model):

    def __init__(self, settings_manager: SettingsManager):
        super(TreeModel, self).__init__()
        self.settings = settings_manager
        LayerInterface.settings = settings_manager
        self.tree_layer = TreeLayer("tree_layer")
        structure_3d_layers = settings_manager("3d_layers/structure")
        self._3d_layers = []
        self._tree_middle_layer = Conv3DLayer(512)
        amount_of_filters_for_3d_layer = settings_manager("3d_layers/amount_of_filters")

        for mode in structure_3d_layers:
            if mode == 0:
                self._3d_layers.append(Conv3DPlaneLayer(amount_of_filters_for_3d_layer))
            elif mode == 1:
                self._3d_layers.append(Inception3DLayer(amount_of_filters_for_3d_layer, 3))
            else:
                raise Exception(f"This structure mode is unknown: {mode}")
            self._3d_layers.append(tf.keras.layers.ReLU())
        #self._3d_layers.append(Inception3DLayer(256, 3))
        #self._3d_layers.append(tf.keras.layers.ReLU())
        self._3d_layers.append(Conv3DLayer(512))

    def print_summary(self):
        def print_fn(text: str):
            if "multiple" in text:
                eles = [e.strip() for e in text.split(" ") if e.strip()]
                if len(eles) > 3 and eles[-1].isnumeric() and (int(eles[-1]) > 0):
                    print(text)
            if "params" in text or "=" in text or "Output Shape" in text:
                print(text)
        self.tree_layer.summary(line_length=120, print_fn=print_fn)
        self.summary(line_length=120, print_fn=print_fn)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.tree_layer(inputs)
        tree_output = self._tree_middle_layer(x)

        for layer in self._3d_layers:
            x = layer(x)
        return x, tree_output

