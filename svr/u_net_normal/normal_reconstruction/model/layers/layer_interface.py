
import tensorflow as tf

from svr.u_net_normal.normal_reconstruction.utility.settings_manager import SettingsManager


class LayerInterface(tf.keras.Model):

    settings: SettingsManager

    def __init__(self, name: str):
        super(LayerInterface, self).__init__(name=name)
        if LayerInterface.settings is None:
            raise Exception("The settings manager of the LayerInterface class has to be set before any child is constructed!")

    def call(self, input_tensor, training=False):
        raise Exception("This function should not be called!")
