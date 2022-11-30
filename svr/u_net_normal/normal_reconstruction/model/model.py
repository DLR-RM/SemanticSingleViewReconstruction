import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, Conv2DTranspose

from svr.u_net_normal.normal_reconstruction.model.layers.inception_layer import InceptionLayer
from svr.u_net_normal.normal_reconstruction.model.layers.layer_interface import LayerInterface
from svr.u_net_normal.normal_reconstruction.utility.settings_manager import SettingsManager


class Model(tf.keras.Model):

    def __init__(self, settings_manager: SettingsManager):
        super(Model, self).__init__()
        self.settings = settings_manager
        LayerInterface.settings = settings_manager
        image_size = self.settings("Training/input_size")
        filter_size = self.settings("Model/normal_filter_size")

        encoder_structure = self.settings("Model/encoder_structure")
        decoder_structure = self.settings("Model/decoder_structure")

        for value in range(5, 9):
            setattr(self, f"encoder_layers_{int(2**value)}", [])
            setattr(self, f"decoder_layers_{int(2 ** value)}", [])
        current_image_size = image_size
        current_used_layer = getattr(self, f"encoder_layers_{current_image_size // 2}")
        for encoder_value in encoder_structure:
            if encoder_value == -1:
                layer = MaxPooling2D()
                current_used_layer.append(layer)
                current_image_size //= 2
                if current_image_size > 32:
                    current_used_layer = getattr(self, f"encoder_layers_{current_image_size // 2}")

            else:
                layer = InceptionLayer(amount_of_filters=encoder_value)
                current_used_layer.append(layer)
                current_used_layer.append(ReLU())

        current_used_layer = getattr(self, f"decoder_layers_{current_image_size}")
        for decoder_value in decoder_structure:
            if decoder_value < 0:
                current_used_layer.append(Conv2DTranspose(-decoder_value, filter_size, strides=(2, 2), padding="same"))
                current_image_size *= 2
                if current_image_size < image_size:
                    current_used_layer = getattr(self, f"decoder_layers_{current_image_size}")
            else:
                layer = InceptionLayer(amount_of_filters=decoder_value)
                current_used_layer.append(layer)
            current_used_layer.append(ReLU())

        current_used_layer.append(Conv2D(3, 3, padding="same"))

    def call(self, input_tensor):
        # encoder
        x = input_tensor
        for layer in self.encoder_layers_256:
            x = layer(x)
        encoder_256_output = x
        for layer in self.encoder_layers_128:
            x = layer(x)
        encoder_128_output = x
        for layer in self.encoder_layers_64:
            x = layer(x)
        encoder_64_output = x
        for layer in self.encoder_layers_32:
            x = layer(x)

        # decoder
        for layer in self.decoder_layers_32:
            x = layer(x)
        # first concat of 64 and this output
        x = tf.concat([x, encoder_64_output], axis=-1)

        for layer in self.decoder_layers_64:
            x = layer(x)
        # first concat of 128 and this output
        x = tf.concat([x, encoder_128_output], axis=-1)

        for layer in self.decoder_layers_128:
            x = layer(x)
        # first concat of 256 and this output
        x = tf.concat([x, encoder_256_output], axis=-1)

        for layer in self.decoder_layers_256:
            x = layer(x)
        return tf.math.l2_normalize(x, axis=-1)


