import math

import tensorflow as tf

from svr.u_net_normal.normal_reconstruction.utility.settings_manager import SettingsManager

class LossManager(object):

    def __init__(self, settings_manager: SettingsManager):
        self.settings = settings_manager
        self.batch_size = self.settings("Training/batch_size")

    @tf.function
    def loss(self, prediction, output):
        loss = self.cosine_loss(output, prediction)
        return 1 - tf.reduce_mean(loss)

    @tf.function
    def cosine_loss(self, prediction, output):
        """
        :param prediction: prediction has to be already normalized
        :param output: output has to be already normalized
        :return: the cosine similarity between the two
        """
        return tf.reduce_sum(output * prediction, axis=-1)

    def angels_percentage(self, prediction, output):
        angels_diff = self.angels_diff_image(prediction, output)
        scalar_values = {}
        for selected_angels in [5.0, 11.5, 22.5, 30.0, 60.0]:
            less = tf.cast(tf.math.less(angels_diff, selected_angels), tf.float32)
            scalar_values[f"angel_diff/{selected_angels}"] = tf.math.reduce_mean(less) * 100.0
        return scalar_values

    @tf.function
    def angels_diff_image(self, prediction, output):
        return tf.math.acos(self.cosine_loss(prediction, output)) * (180.0 / math.pi)
