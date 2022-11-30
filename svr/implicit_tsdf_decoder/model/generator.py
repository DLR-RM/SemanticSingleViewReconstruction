
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense,  BatchNormalization
from tensorflow.keras import Model

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager, NotFoundException

class Generator(Model):

    def __init__(self, settings_manager: SettingsManager):
        super(Generator, self).__init__(self)
        self.settings = settings_manager
        self._trunc_threshold = self.settings("Generator/trunc_threshold")
        self._coords_size = self.settings("Generator/coords_size")
        self._use_batch_norm = self.settings("Generator/use_batch_norm")

        self._mapping_size = self.settings("Generator/fourier_mapping_size")
        self._use_fourier_mapping = self.settings("Generator/fourier_use_mapping")
        self._fourier_mapping_scale = self.settings("Generator/fourier_mapping_scale")
        self._use_class_prediction = self.settings("Generator/use_classes")
        self._number_of_classes = self.settings("Generator/number_of_classes")
        self._final_class_layers = self.settings("Generator/final_class_layers")
        try:
            self.activation_type = self.settings("Generator/activation_type")
            self.sinus_w0_first = self.settings("Generator/sinus_w0_first")
            self.sinus_w0_hidden = self.settings("Generator/sinus_w0_hidden")
        except NotFoundException:
            self.activation_type = "RELU"

        self._used_layer_amounts = self.settings("Generator/layers")
        used_concats = self.settings("Generator/concats")
        if not isinstance(self._used_layer_amounts, list):
            raise Exception(f"The layers are not saved as a list, which is required: {self._used_layer_amounts}!")
        if len(used_concats) != len(self._used_layer_amounts):
            raise Exception(f"The length of the used concats is not equal to the used layers: "
                            f"{used_concats}, {self._used_layer_amounts}")
        fac = 1
        self._class_layers = []
        self._layers = []
        w0_value = self.sinus_w0_first
        for layer_nr, (dense_amount, use_concat) in enumerate(zip(self._used_layer_amounts, used_concats)):
            dense_amount *= fac
            if use_concat:
                self._layers.append(tf.keras.layers.Concatenate(axis=1))
            if self.activation_type.lower().strip() == "relu":
                self._layers.append(Dense(dense_amount, activation="relu", use_bias=True, name='fc_{}_{}'.format(dense_amount, layer_nr)))
            elif self.activation_type.lower().strip() == "siren":
                from tf_siren import SinusodialRepresentationDense
                self._layers.append(SinusodialRepresentationDense(dense_amount, w0=w0_value, name='siren_fc_{}_{}'.format(dense_amount, layer_nr)))
            else:
                raise Exception(f"The activation type is unknown: {self.activation_type}")
            w0_value = self.sinus_w0_hidden
            if self._use_batch_norm:
                self._layers.append(BatchNormalization())
        # create the output layer
        dense_amount = 1
        layer_nr = "last"

        self._amount_of_layers = len(self._used_layer_amounts)
        self._layers.append(Dense(dense_amount, use_bias=True, name='fc_{}_{}'.format(dense_amount, layer_nr)))

        # create the class layers
        if self._use_class_prediction:
            for layer_nr, dense_amount in enumerate(self._final_class_layers):
                dense_amount *= fac
                self._class_layers.append(Dense(dense_amount, activation="relu", use_bias=True, name='fc_{}_{}'.format(dense_amount, layer_nr + len(self._used_layer_amounts))))
                if self._use_batch_norm:
                    self._class_layers.append(BatchNormalization())
            self._class_layers.append(Dense(self._number_of_classes, activation="softmax", use_bias=True, name='predictions_{}_{}'.format(dense_amount, "last_class")))

        if self._use_fourier_mapping:
            # variable for new fourier mapping
            # used random variable
            prng = np.random.RandomState(0)
            fourier_mapping = prng.normal(scale=self._fourier_mapping_scale, size=(self._mapping_size, self._coords_size)).transpose()
            self.fourier_mapping = tf.constant(fourier_mapping, dtype=tf.float32)

    def calc_amount_of_operations(self):
        """
        Returns the amount of operations necessary for each iteration. (Multiplications, Adds)

        :return: Tuple(int, int): Multiplications, Adds
        """
        latent_size = self.settings("Generator/latent_dim")
        if self._use_fourier_mapping:
            input_size = latent_size + self._mapping_size * 2
        else:
            input_size = latent_size + self._coords_size
        current_size = input_size
        additions = 0
        multiplications = 0
        counter = 0
        for layer in self._layers:
            if isinstance(layer, tf.keras.layers.Dense):
                output_size = layer.units
                multiplications += output_size * current_size
                additions += (current_size - 1) * output_size + output_size
            elif isinstance(layer, tf.keras.layers.Concatenate):
                output_size = current_size + input_size
            else:
                raise Exception("Layer type is unknown: {}".format(type(layer)))
            counter += 1
            if counter == self._amount_of_layers:
                last_layer_before_end_size = output_size
            current_size = output_size
        if self._use_class_prediction:
            current_size = last_layer_before_end_size
            for c_layer in self._class_layers:
                output_size = c_layer.units
                multiplications += output_size * current_size
                additions += (current_size - 1) * output_size + output_size
                current_size = output_size
        return (multiplications, additions)

    def input_mapping(self, x):
        """
        https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
        :param x:
        :return:
        """
        x_proj = (tf.constant(2. * np.pi) * x) @ self.fourier_mapping
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

    def prepare_input(self, inputs):
        # extract the coord values, this way we can add them several times in the network, to improve their importance
        coord_input = inputs[:, :self._coords_size]
        if self._use_fourier_mapping:
            fourier_coords = self.input_mapping(coord_input)
            # add the fourier coords with the latent_values
            x = tf.concat([fourier_coords, inputs[:, self._coords_size:]], axis=-1)
            return x, fourier_coords
        return inputs, coord_input

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x, coord_input = self.prepare_input(inputs)

        counter = 0
        for layer in self._layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                x = layer([x, coord_input])
            else:
                x = layer(x)
            counter += 1
            if counter == self._amount_of_layers:
                last_layer_before_end = x
        x = tfp.math.clip_by_value_preserve_gradient(x, -self._trunc_threshold, self._trunc_threshold)

        if self._use_class_prediction:
            class_x = last_layer_before_end
            for c_layer in self._class_layers:
                class_x = c_layer(class_x)
            x = tf.concat([x, class_x], axis=-1)
        return x

    def calling(self, inputs):
        x, coord_input = self.prepare_input(inputs)

        for layer in self._layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                x = layer([x, coord_input])
            else:
                x = layer(x)
        x = tfp.math.clip_by_value_preserve_gradient(x, -self._trunc_threshold, self._trunc_threshold)
        return x
