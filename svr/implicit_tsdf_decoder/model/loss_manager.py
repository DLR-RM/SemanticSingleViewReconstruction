

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import Loss

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager

class LossManager(Loss):

    def __init__(self, settings_manager: SettingsManager): #trunc_threshold, add_surface_weights=True, add_corner_weights=True, add_sign_weights=True):
        Loss.__init__(self)
        self.settings = settings_manager
        self._truncation_threshold = self.settings("Generator/trunc_threshold")  # set in the DataSetLoader
        self.add_surface_weights = self.settings("LossManager/add_surface_weights")
        self.add_corner_weights = self.settings("LossManager/add_corner_weights")
        self.add_sign_weights = self.settings("LossManager/add_sign_weights")
        self._surface_loss_weight = self.settings("LossManager/surface_loss_weight")
        self._use_class_predictions = self.settings("Generator/use_classes")
        self._output_size = self.settings("Generator/output_size")
        self._number_of_classes = self.settings("Generator/number_of_classes")
        self._class_weighting_loss = self.settings("Generator/class_weighting_loss")
        self._surface_loss_type = self.settings("LossManager/surface_loss_type")

        self._boundary_selection_scale = self.settings("DataLoader/boundary_selection_scale")
        self._boundary_selection_scale_half = 0
        if self._boundary_selection_scale != 0:
            self._boundary_selection_scale_half = (self._boundary_selection_scale - 1.0) * 0.5 + 1.0

        if self._surface_loss_type.upper() == "GAUSS":
            self.surface_weights_loss = self.gauss_surface_weights_loss
        elif self._surface_loss_type.upper() == "EXP":
            self.surface_weights_loss = self.exp_surface_weights_loss
        else:
            raise Exception(f"Unknown surface loss type: {self._surface_loss_type}")

        self._categorical_loss = tf.keras.losses.CategoricalCrossentropy()
        self._categorical_acc = tf.keras.metrics.CategoricalAccuracy()
        self._default_dist = tfp.distributions.Normal(loc=0.0, scale=self._truncation_threshold / 4)
        self._neg_dist = tfp.distributions.Normal(loc=0.0, scale=self._truncation_threshold / 5)
        self._lat_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.summaries = []


    # Get the normal pdf for the given value
    @staticmethod
    def get_norm_pdf_value(dist, value):
        """ To get the normal pdf for the given value"""
        return dist.prob(value)

    @tf.function
    def diff(self, y_true, y_pred):
        tsdf_y_true, tsdf_y_pred, classes_y_true, classes_y_pred = self.extract_values(y_true, y_pred)
        return tf.abs(tf.subtract(tsdf_y_true, tsdf_y_pred))

    @tf.function
    def class_loss(self, y_true, y_pred):
        tsdf_y_true, tsdf_y_pred, classes_y_true, classes_y_pred = self.extract_values(y_true, y_pred)
        return self._categorical_loss(classes_y_true, classes_y_pred)

    @tf.function
    def class_accuracy(self, y_true, y_pred):
        self._categorical_acc.reset_state()
        tsdf_y_true, tsdf_y_pred, classes_y_true, classes_y_pred = self.extract_values(y_true, y_pred)
        return self._categorical_acc(classes_y_true, classes_y_pred)

    @tf.function
    def extract_values(self, y_true, y_pred):
        tsdf_y_pred = y_pred[:, :self._output_size]
        tsdf_y_true = y_true[:, :self._output_size]
        if self._use_class_predictions:
            classes_y_pred = y_pred[:, self._output_size: self._output_size + self._number_of_classes]
            classes_y_true = y_true[:, self._output_size: self._output_size + self._number_of_classes]
        else:
            classes_y_true, classes_y_pred = None, None
        return tsdf_y_true, tsdf_y_pred, classes_y_true, classes_y_pred

    @tf.function
    def exp_surface_weights_loss(self, tsdf_y_true, difference):
        return difference * ((self._surface_loss_weight * self._truncation_threshold) / (tf.abs(tsdf_y_true) + 1e-3))

    @tf.function
    def gauss_surface_weights_loss(self, tsdf_y_true, difference):
        # Give large weights for the losses near the surface weights are
        # proportional to the pdf for the sdf value at that point

        default_surface_weights = self.get_norm_pdf_value(self._default_dist, tsdf_y_true)
        return difference * (default_surface_weights * 4.0 / self._truncation_threshold)

        # self.summaries.append(tf.summary.histogram("Default surface weight", default_surface_weights))

        neg_weights = self.get_norm_pdf_value(self._neg_dist, tsdf_y_true)
        neg_weights = tf.cast(tf.less_equal(tsdf_y_true, 0), tf.float32) * neg_weights
        # self.summaries.append(tf.summary.histogram("Neg surface weight", neg_weights))

        surface_weights = tf.add(default_surface_weights, neg_weights)

        densities = self._surface_loss_weight * tf.multiply(surface_weights, 6 * self._truncation_threshold)
        return densities * difference

    @tf.function
    def sign_weights_loss(self, tsdf_y_pred, tsdf_y_true):
        gen_out_sign = tf.math.sign(tsdf_y_pred)
        data_sign = tf.math.sign(tsdf_y_true)
        abs_diff_sign = tf.abs(gen_out_sign - data_sign)
        return abs_diff_sign


    @tf.function
    def corner_weights_loss(self, coord_input, diff):
        """
        Calculates the corner weight loss, by first calculating the min and max coordinate of each point.

        :param coord_input:
        :param diff:
        :return:
        """
        # calculate min and max coordinate of each point
        min_coord = tf.math.reduce_min(coord_input, axis=-1, keepdims=True)
        max_coord = tf.math.reduce_max(coord_input, axis=-1, keepdims=True)
        # calculate distance to boundary at -1 and 1
        if self._boundary_selection_scale != 0.0:
            min_dist = min_coord - (-1.0 * self._boundary_selection_scale_half)
            max_dist = 1.0 * self._boundary_selection_scale_half - max_coord
        else:
            min_dist = min_coord - (-1.0)
            max_dist = 1.0 - max_coord
        # concat the distances and absolute them and find if max or min dist is closer by reduce_min -> invert number
        # so that small values are big
        abs_max_dist = 1.0 - tf.reduce_min(tf.abs(tf.concat([min_dist, max_dist], axis=-1)), axis=-1, keepdims=True)
        # square them for a sharper result
        squared_abs_max_dist = tf.square(abs_max_dist)
        return diff * squared_abs_max_dist

    @tf.function
    def __call__(self, y_true, y_pred, coord_input):
        """
        Calculates the reconstruction loss between the generated data and ground truth

        :param y_true: The ground truth
        :param y_pred: The prediction truth
        :param coord_input: The coordinate input [batch_size, coords_size], coords_size is usually 3
        """

        tsdf_y_true, tsdf_y_pred, classes_y_true, classes_y_pred = self.extract_values(y_true, y_pred)

        org_diff = self.diff(tsdf_y_true, tsdf_y_pred)
        diff = org_diff

        if self.add_surface_weights:
            diff += self.surface_weights_loss(tsdf_y_true, org_diff)

            #self.summaries.append(tf.summary.histogram("Total surface weight", densities))
            #self.summaries.append(tf.summary.scalar("Total surface weight loss", tf.reduce_mean(diff * densities)))


        if self.add_sign_weights:
            diff += self.sign_weights_loss(tsdf_y_pred, tsdf_y_true)
            #self.summaries.append(tf.summary.histogram("Sign surface weight", abs_diff_sign))
            #self.summaries.append(tf.summary.scalar("Sign surface weight loss", tf.reduce_mean(diff * abs_diff_sign)))

        if self.add_corner_weights:
            diff += self.corner_weights_loss(coord_input, org_diff)

        self.not_reduced_loss = diff
        final_loss = tf.reduce_mean(self.not_reduced_loss)
        if self._use_class_predictions:
            class_loss = self._categorical_loss(classes_y_true, classes_y_pred)
            final_loss += self._class_weighting_loss * class_loss

        return final_loss

