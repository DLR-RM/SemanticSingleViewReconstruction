import tensorflow as tf

from svr.scene_reconstruction.utility.settings_manager import SettingsManager

class LossManager(object):

    def __init__(self, settings_manager: SettingsManager):
        self.settings = settings_manager
        self.inner_tree_loss_weight = self.settings("TreeModel/inner_tree_loss_weight")
        self._use_loss_shaping = self.settings("LossManager/use_loss_shaping")
        self.batch_size = self.settings("Training/batch_size")

    @tf.function
    def loss(self, prediction, tree_prediction, output, loss_factor):
        loss_end = tf.reduce_mean(self.loss_single(prediction, output, loss_factor))
        loss_tree = tf.reduce_mean(self.loss_single(tree_prediction, output, loss_factor))
        loss = loss_end + loss_tree * self.inner_tree_loss_weight
        return loss

    @tf.function
    def loss_single(self, prediction, output, loss_factor):
        difference = tf.reduce_mean(self.difference(prediction, output), axis=-1)
        if self._use_loss_shaping:
            loss = difference * loss_factor
            return loss
        else:
            return difference

    @staticmethod
    @tf.function
    def difference(prediction, output):
        return tf.abs(output - prediction)
