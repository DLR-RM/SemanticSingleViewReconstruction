import logging
import time
import os

import tensorflow as tf
import numpy as np

from svr.implicit_tsdf_decoder.model.generator import Generator
from svr.implicit_tsdf_decoder.model.loss_manager import LossManager
from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager


class Trainer(object):

    def __init__(self, settings_manager: SettingsManager):

        self.settings = settings_manager

        self.batch_size = self.settings("Training/batch_size")
        self.point_amount = self.settings("Training/point_amount")
        self.latent_summary_steps = self.settings("Training/summary_steps")
        self.gen_summary_steps = self.settings("Training/summary_steps")
        if self.settings("Training/latent_learning_rate_mode").lower() == "step_decay":
            self.latent_summary_steps = self.settings("Training/latent_learning_drop_epoch_amount")
            self.latent_steps = int(self.settings("Training/latent_steps") // self.latent_summary_steps)
        else:
            self.latent_steps = int(self.settings("Training/latent_steps") // self.latent_summary_steps)
        self.gen_steps = int(self.settings("Training/gen_steps") // self.gen_summary_steps)
        self.latent_dim = self.settings("Generator/latent_dim")
        self.coords_size = self.settings("Generator/coords_size")
        self.output_size = self.settings("Generator/output_size")
        self.number_of_classes = self.settings("Generator/number_of_classes")
        self.tsdf_threshold = self.settings("Generator/trunc_threshold")
        self.use_gradient_smoothing = self.settings("Training/use_gradient_smoothing")
        self.gradient_size = self.settings("Training/gradient_size")
        self.gradient_loss_scaling = self.settings("Training/gradient_loss_scaling")
        self.use_classes = self.settings("Generator/use_classes")

        # latent variable has the same dim as the amount of reconstructed volumes at the same time
        # most of them won't be used though, as a lot of id sets are similar
        self.latent_variable = tf.Variable(tf.random.normal([self.batch_size, self.latent_dim]), trainable=True,
                                           name="latent_variable")

        # variable which holds the current coordinates
        self.coord_var = tf.Variable(tf.zeros([self.batch_size, self.point_amount, self.coords_size +
                                               self.output_size + self.number_of_classes]), trainable=False, name="Coord_var")
        # variable which hold the current selected used ids
        self.used_ids_var = tf.Variable(tf.zeros([self.batch_size, 1], dtype=tf.int32),
                                        name="Used_ids_var", dtype=tf.int32)

        # variable which holds the current coordinates
        self.model_input_var = tf.Variable(tf.zeros([self.batch_size * self.point_amount, self.coords_size +
                                                     self.output_size + self.number_of_classes + self.latent_dim], dtype=tf.float32),
                                           trainable=False, name="Model_input_var")

        # create the Generator
        self.gen: Generator = Generator(self.settings)
        self.loss_manager = LossManager(self.settings)
        self.latent_learning_rate = float(self.settings("Training/latent_learning_rate"))
        self.latent_optimizer = tf.keras.optimizers.Adam(learning_rate=self.latent_learning_rate)
        self.gen_optimizer = tf.keras.optimizers.Adam()

        # variables for the hyperbolic space
        self.use_hyperbolic_space = False
        # curvature of the hyperbolic space
        self.c = 1.0
        self.sqrt_c = np.sqrt(self.c)
        self.max_norm = (1 - 1e-3) / self.sqrt_c
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

    @tf.function
    def setup_variables(self, points_input, output_tsdf, classes, ids, latent_vec=None):
        # reset latent_variable
        if latent_vec is None:
            self.latent_variable.assign(tf.zeros([self.batch_size, self.latent_dim]))
            # reset latent adam optimizer
            for var in self.latent_optimizer.variables():
                var.assign(tf.zeros_like(var))
        else:
            self.latent_variable.assign(latent_vec)

        #self.latent_variable.assign(tf.random.normal([self.batch_size, self.latent_dim], stddev=0.01))
        # assign the used ids
        class_on_hot_encoded = tf.one_hot(classes, depth=self.number_of_classes, on_value=1.0, off_value=0.0, axis=-1)
        # first output_tsdf, then all classes, and then the coords
        coord_combined = tf.concat([tf.expand_dims(output_tsdf, axis=2), class_on_hot_encoded, points_input], axis=2)
        self.coord_var.assign(coord_combined)

        unique_id_collection, used_ids = tf.unique(ids[:, 0])
        used_ids = tf.reshape(used_ids, [self.batch_size, 1])
        self.used_ids_var.assign(used_ids)

    @tf.function
    def get_current_latent_values(self):
        #if self.use_hyperbolic_space:
        #    sqrt_c = self.sqrt_c
        #    max_norm = self.max_norm

        #    def tanh(x, clamp=15):
        #        return tf.tanh(tf.clip_by_value(x, -clamp, clamp))

        #    def _lambda_x(x, c, keepdim: bool = False):
        #        return 2 / (1 - c * tf.reduce_sum(tf.math.pow(x, 2), axis=-1, keepdims=keepdim))

        #    def _mobius_add(x, y, c):
        #        x2 = tf.reduce_sum(tf.math.pow(x, 2), axis=-1, keepdims=True)
        #        y2 = tf.reduce_sum(tf.math.pow(y, 2), axis=-1, keepdims=True)
        #        xy = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        #        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        #        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        #        return num / (denom + 1e-5)

        #    def _expmap(x, u, c):  # pragma: no cover
        #        u_norm = tf.clip_by_value(tf.norm(u, ord=2, axis=-1, keepdims=True), 1e-5, tf.float32.max)
        #        second_term = (tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True) * u_norm) * u /
        #                       (sqrt_c * u_norm))
        #        gamma_1 = _mobius_add(x, second_term, c)
        #        return gamma_1

        #    def expmap0(u):
        #        u_norm = tf.clip_by_value(tf.norm(u, ord=2, axis=-1, keepdims=True), 1e-5, tf.float32.max)
        #        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        #        return gamma_1

        #    def project(x):
        #        norm = tf.clip_by_value(tf.norm(x, ord=2, axis=-1, keepdims=True), 1e-5, tf.float32.max)
        #        latent_var = x / norm * max_norm
        #        return tf.where(x > max_norm, latent_var, x)

        #    xp_latent_var = project(expmap0(self.latent_variable))
        #    used_latent_var = project(_expmap(xp_latent_var, self.latent_variable, self.c))
        #    used_latent_variables = tf.gather_nd(used_latent_var, self.used_ids_var)
        #else:
        used_latent_variables = tf.gather_nd(self.latent_variable, self.used_ids_var)
        used_latent_variables = tf.reshape(used_latent_variables, [self.batch_size, 1, self.latent_dim])
        used_latent_variables = tf.tile(used_latent_variables, [1, self.point_amount, 1])
        return used_latent_variables

    @tf.function
    def get_current_model_input_live(self):
        """
        This functions returns the model_input as one big matrix, this should only be executed with care.
        :return:
        """
        used_latent_variables = self.get_current_latent_values()

        # combine batches with points
        reshaped_coord_var = tf.reshape(self.coord_var, [self.batch_size * self.point_amount,
                                                         self.output_size + self.coords_size + self.number_of_classes])
        reshaped_latent_var = tf.reshape(used_latent_variables, [self.batch_size * self.point_amount, self.latent_dim])

        model_input = tf.concat([reshaped_coord_var, reshaped_latent_var], axis=-1)
        return model_input

    @tf.function
    def update_current_inputs(self):
        model_input = self.get_current_model_input_live()
        self.model_input_var.assign(model_input)

    @tf.function
    def get_current_inputs(self):
        used_output = self.model_input_var[:, :self.output_size + self.number_of_classes]
        coord_input = self.model_input_var[:, self.output_size + self.number_of_classes: self.output_size + self.number_of_classes + self.coords_size]
        classes_input = self.model_input_var[:, self.output_size: self.output_size + self.number_of_classes]
        latent_input = self.model_input_var[:, self.output_size + self.number_of_classes + self.coords_size:]
        input_to_gen = self.model_input_var[:, self.output_size + self.number_of_classes:]
        return used_output, coord_input, latent_input, classes_input, input_to_gen

    def get_current_inputs_live(self):
        model_input = self.get_current_model_input_live()
        used_output = model_input[:, :self.output_size + self.number_of_classes]
        coord_input = model_input[:, self.output_size + self.number_of_classes:self.output_size + self.number_of_classes + self.coords_size]
        classes_input = model_input[:, self.output_size: self.output_size + self.number_of_classes]
        latent_input = model_input[:, (self.output_size + self.coords_size + self.number_of_classes):]
        input_to_gen = model_input[:, self.output_size + self.number_of_classes:]
        return used_output, coord_input, latent_input, classes_input, input_to_gen

    def init_optimize_latent_code(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.latent_variable)
            used_output, coord_input, _, _, input_to_gen = self.get_current_inputs_live()
            predictions = self.gen(input_to_gen, training=True)
            loss = self.loss_manager(used_output, predictions, coord_input)
        gradients = tape.gradient(loss, self.latent_variable)
        self.latent_optimizer.apply_gradients(zip([gradients], [self.latent_variable]))

    @tf.function
    def optimize_latent_code(self):
        #res_gradients = []
        for _ in tf.range(self.latent_summary_steps):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.latent_variable)
                used_output, coord_input, _, _, input_to_gen = self.get_current_inputs_live()
                predictions = self.gen(input_to_gen, training=True)
                loss = self.loss_manager(used_output, predictions, coord_input)
            gradients = tape.gradient(loss, self.latent_variable)
            #self.copied_latent_var.assign(self.latent_variable)
            self.latent_optimizer.apply_gradients(zip([gradients], [self.latent_variable]))
            #res_gradients.append(tf.abs(self.latent_variable - self.copied_latent_var))
        #return tf.reduce_sum(res_gradients, axis=0)

    @tf.function
    def train(self):
        if self.use_gradient_smoothing:
            used_output, coord_input, latent_input, _, input_to_gen = self.get_current_inputs()
            for _ in tf.range(self.gen_summary_steps):
                with tf.GradientTape() as tape:
                    with tf.GradientTape(watch_accessed_variables=False) as tape2:
                        tape2.watch(input_to_gen)
                        predictions = self.gen(input_to_gen, training=True)
                        loss = self.loss_manager(used_output, predictions, coord_input)
                    output_gradient = tf.norm(tape2.gradient(predictions, input_to_gen)[:, :self.coords_size], axis=-1)
                    loss_on_gradient = tf.reduce_mean(tf.nn.relu(output_gradient - self.gradient_size))
                    loss += self.gradient_loss_scaling * loss_on_gradient
                gradients = tape.gradient(loss, self.gen.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        else:
            used_output, coord_input, latent_input, _, input_to_gen = self.get_current_inputs()
            for _ in tf.range(self.gen_summary_steps):
                with tf.GradientTape() as tape:
                    predictions = self.gen(input_to_gen, training=True)
                    loss = self.loss_manager(used_output, predictions, coord_input)
                gradients = tape.gradient(loss, self.gen.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))

    @tf.function
    def loss_value(self):
        if self.use_gradient_smoothing:
            used_output = self.model_input_var[:, :self.output_size + self.number_of_classes]
            coord_input = self.model_input_var[:, self.output_size + self.number_of_classes:self.output_size + self.coords_size + self.number_of_classes]
            input_to_gen = self.model_input_var[:, self.output_size + self.number_of_classes:]
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(input_to_gen)
                predictions = self.gen(input_to_gen, training=True)
                loss = self.loss_manager(used_output, predictions, coord_input)
            output_gradient = tf.norm(tape2.gradient(predictions, input_to_gen)[:, :self.coords_size], axis=-1)
            loss_on_gradient = tf.reduce_mean(tf.nn.relu(output_gradient - self.gradient_size))
            loss += self.gradient_loss_scaling * loss_on_gradient
            return loss
        else:
            used_output = self.model_input_var[:, :self.output_size + self.number_of_classes]
            coord_input = self.model_input_var[:, self.output_size + self.number_of_classes:self.output_size + self.coords_size + self.number_of_classes]
            input_to_gen = self.model_input_var[:, self.output_size + self.number_of_classes:]
            predictions = self.gen(input_to_gen, training=False)
            return self.loss_manager(used_output, predictions, coord_input)

    @tf.function
    def sum_loss_value(self):
        """
        Returns interesting function scores, are saved in a dict:
        1. diff_total: total difference between target and prediction
        2. diff_on_the_edge: difference in the edge region defined by the target, abs(y) < 10% * threshold
        3. mean_iou: iou over prediction and target
        4. loss on the gradient smoothing, None if not used
        """
        results = {}
        used_output, coord_input, latent_input, _, input_to_gen = self.get_current_inputs()
        if self.use_gradient_smoothing:
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(input_to_gen)
                predictions = self.gen(input_to_gen, training=False)
            output_gradient = tf.norm(tape2.gradient(predictions, input_to_gen)[:, :self.coords_size], axis=-1)

            results["loss_on_gradient_histo"] = output_gradient
            results["loss_on_gradient"] = tf.reduce_mean(tf.nn.relu(output_gradient - self.gradient_loss_scaling)) * self.gradient_loss_scaling
        else:
            predictions = self.gen(input_to_gen, training=False)
        diff = self.loss_manager.diff(used_output, predictions)
        diff_total = tf.reduce_mean(diff)
        results["diff_total"] = diff_total

        tsdf_y_true, tsdf_y_pred, _, _ = self.loss_manager.extract_values(used_output, predictions)
        # reduce the error outside of the edge to zero
        diff = tf.where(tf.greater(tf.abs(tsdf_y_true), 0.1 * self.tsdf_threshold), diff, 0)
        diff_on_edge = tf.reduce_mean(diff)
        results["diff_on_edge"] = diff_on_edge

        if self.loss_manager.add_surface_weights:
            results["surface_loss"] = tf.reduce_mean(self.loss_manager.surface_weights_loss(tsdf_y_true, diff))
        if self.loss_manager.add_sign_weights:
            results["sign_loss"] = tf.reduce_mean(self.loss_manager.sign_weights_loss(tsdf_y_pred, tsdf_y_true))
        if self.loss_manager.add_corner_weights:
            results["corner_loss"] = tf.reduce_mean(self.loss_manager.corner_weights_loss(coord_input, diff))

        self.iou_metric.reset_states()
        self.iou_metric.update_state(tsdf_y_true <= 0, tsdf_y_pred <= 0)
        results["iou"] = self.iou_metric.result()

        for coverage in [85, 90, 95, 98]:
            self.iou_metric.reset_states()
            changed_tsdf_y_true = tf.where(tf.greater(tf.abs(tsdf_y_true), (1.0 - coverage / 100.0) * self.tsdf_threshold), tsdf_y_true, 0)
            changed_tsdf_y_pred = tf.where(tf.greater(tf.abs(tsdf_y_true), (1.0 - coverage / 100.0) * self.tsdf_threshold), tsdf_y_pred, 0)
            self.iou_metric.update_state(changed_tsdf_y_true <= 0, changed_tsdf_y_pred <= 0)
            results[f"iou_{coverage}_percent"] = self.iou_metric.result()

        if self.use_classes:
            results["classes"] = self.loss_manager.class_loss(used_output, predictions)
            results["class_accuracy"] = self.loss_manager.class_accuracy(used_output, predictions)

        return results

    @tf.function
    def loss_tensor(self):
        used_output, coord_input, latent_input, _, input_to_gen = self.get_current_inputs()
        predictions = self.gen(input_to_gen, training=False)
        return self.loss_manager.diff(used_output, predictions)

    @tf.function
    def predict_current_inputs_live(self):
        _, _, _, _, input_to_gen = self.get_current_inputs_live()
        predictions = self.gen(input_to_gen, training=False)
        return predictions

    def perform_latent_opt(self, summary_update, verbose=True):
        if verbose and self.latent_steps > 1:
            start_time = time.time()
            logging.info(f"Current loss is: {self.loss_value().numpy()}")
            progbar = tf.keras.utils.Progbar(self.latent_steps - 1)
        for i in range(self.latent_steps):
            # this update is only done once per optimizing step it saves the current model_input information inside a
            # variable -> this is needed here for the summaries
            if summary_update:
                self.update_current_inputs()
                summary_update(self, i)
            self.optimize_latent_code()
            self.set_latent_optimizer_learning_rate(i)
            if verbose and self.latent_steps > 1:
                progbar.update(i)
        if verbose:
            self.update_current_inputs()
            results = self.sum_loss_value()
            info_stream = f"After {self.latent_steps * self.latent_summary_steps} opt steps, loss is: {self.loss_value().numpy()}, " \
                          f"loss diff: {results['diff_total'].numpy()}, loss on edge: {results['diff_on_edge'].numpy()}, " \
                          f"IoU: {results['iou'].numpy()}"
            if self.use_gradient_smoothing:
                info_stream += f", loss gradient: {results['loss_on_gradient'].numpy()}"
            if self.use_classes:
                info_stream += f", classes loss: {results['classes'].numpy()}, class acc: {results['class_accuracy'] * 100.0}%"
            info_stream += f", took: {time.time() - start_time}"
            logging.info(info_stream)

    def perform_latent_opt_early_stopping(self):
        #self.accumulated_gradient = []
        for i in range(self.latent_steps):
            # this update is only done once per optimizing step it saves the current model_input information inside a
            # variable -> this is needed here for the summaries
            self.optimize_latent_code()
            #self.accumulated_gradient.append(gradient)
            if i < 2:
                continue
            used_output, _, _, _, input_to_gen = self.get_current_inputs_live()
            predictions = self.gen(input_to_gen, training=False)
            tsdf_y_true, tsdf_y_pred, _, _ = self.loss_manager.extract_values(used_output, predictions)
            diff = self.loss_manager.diff(used_output, predictions)
            diff_total = tf.reduce_mean(diff)
            if self.use_classes:
                class_accuracy = self.loss_manager.class_accuracy(used_output, predictions)
                if diff_total < 0.02 and class_accuracy > 0.95:
                    return i + 1
            else:
                if diff_total < 0.02:
                    return i + 1
        return self.latent_steps

    def perform_latent_opt_early_stopping_non_boundary(self):
        for i in range(self.latent_steps):
            # this update is only done once per optimizing step it saves the current model_input information inside a
            # variable -> this is needed here for the summaries
            self.optimize_latent_code()
            if i < 2:
                continue
            used_output, _, _, _, input_to_gen = self.get_current_inputs_live()
            predictions = self.gen(input_to_gen, training=False)
            tsdf_y_true, tsdf_y_pred, _, _ = self.loss_manager.extract_values(used_output, predictions)
            diff = self.loss_manager.diff(used_output, predictions)
            # reduce the error outside of the edge to zero
            diff = tf.where(tf.greater(tf.abs(tsdf_y_true), 0.1 * self.tsdf_threshold), diff, 0)
            diff_on_edge = tf.reduce_mean(diff)
            if diff_on_edge < 0.03:
                return i + 1
        return self.latent_steps

    def perform_gen_training(self, summary_update, verbose=True):
        start_time = time.time()
        # this update is only done once per training step it saves the current model_input information inside a
        # variable
        self.update_current_inputs()
        self.set_gen_optimizer_learning_rate()
        progbar = tf.keras.utils.Progbar(self.gen_steps - 1)
        for i in range(self.gen_steps):
            summary_update(self, i)
            self.train()
            progbar.update(i)
        if verbose:
            results = self.sum_loss_value()
            info_stream = f"Generator training step {self.gen_steps * self.gen_summary_steps} done, loss: {self.loss_value().numpy()}, " \
                          f"loss diff: {results['diff_total'].numpy()}, loss on edge: {results['diff_on_edge'].numpy()}, " \
                          f"IoU: {results['iou'].numpy()}"
            if self.use_gradient_smoothing:
                info_stream += f", loss gradient: {results['loss_on_gradient'].numpy()}"
            if self.use_classes:
                info_stream += f", classes loss: {results['classes'].numpy()}, class acc: {results['class_accuracy'] * 100.0}%"
            info_stream += f", took: {time.time() - start_time}"
            logging.info(info_stream)

    def set_latent_optimizer_learning_rate(self, latent_step: int):
        latent_step *= self.latent_summary_steps
        optimizer = self.latent_optimizer
        if self.settings("Training/latent_learning_rate_mode").lower() == "fixed":
            optimizer.lr = float(self.settings("Training/latent_learning_rate"))
        elif self.settings("Training/latent_learning_rate_mode").lower() == "step_decay":
            drop = self.settings("Training/latent_learning_drop_decay")
            learning_rate = float(self.settings("Training/latent_learning_rate")) * tf.math.pow(drop, tf.math.floor((1.0+latent_step)/self.latent_summary_steps))
            optimizer.lr = learning_rate
        else:
            raise Exception("Unknown latent learning rate mode")

    def set_gen_optimizer_learning_rate(self):
        self.gen_optimizer.lr = float(self.settings("Training/gen_learning_rate"))

    def add_loss_summaries_to_writer(self, writer, counter: int):
        with writer.as_default():
            results = self.sum_loss_value()
            tf.summary.scalar('loss_diff', results['diff_total'], step=counter)
            tf.summary.scalar('loss_diff_on_edge', results["diff_on_edge"], step=counter)
            tf.summary.scalar('mean_iou', results["iou"], step=counter)
            for iou_coverage in [85, 90, 95, 98]:
                used_key = f"iou_{iou_coverage}_percent"
                if used_key in results:
                    tf.summary.scalar(f"mean_iou_{iou_coverage}_percent", results[used_key], step=counter)
            tf.summary.scalar('loss', self.loss_value(), step=counter)
            tf.summary.histogram('loss_histo', tf.clip_by_value(self.loss_tensor(), 0, 0.1), step=counter)
            tf.summary.histogram('latent_variable', self.latent_variable, step=counter)
            if "classes" in results:
                tf.summary.scalar('classes', results["classes"], step=counter)
                tf.summary.scalar('scaled_classes', results["classes"] * self.loss_manager._class_weighting_loss, step=counter)
                tf.summary.scalar('class_accuracy', results["class_accuracy"], step=counter)
            if "loss_gradient" in results:
                tf.summary.scalar('loss_gradient_smoothing', results["loss_on_gradient"], step=counter)
            if "loss_gradient_histo" in results:
                tf.summary.histogram('loss_gradient_smoothing_histo', results["loss_on_gradient_histo"], step=counter)

            if "surface_loss" in results:
                tf.summary.scalar('surface_loss', results["surface_loss"], step=counter)
            if "sign_loss" in results:
                tf.summary.scalar('sign_loss', results["sign_loss"], step=counter)
            if "corner_loss" in results:
                tf.summary.scalar('corner_loss', results["corner_loss"], step=counter)
            tf.summary.scalar('loss', self.loss_value(), step=counter)

    def save_weights(self, counter):
        logging.info("Save weights")
        checkpoint_path = os.path.join(self.settings.base_log_dir, "cp-{epoch:04d}.ckpt")
        self.gen.save_weights(checkpoint_path.format(epoch=counter))
        opt_checkpoint_path = os.path.join(self.settings.base_log_dir, "cp-opt-gen-{epoch:04d}.ckpt")
        # weights for latent opt should be zero at start of each optimizing -> so no saving
        np.save(opt_checkpoint_path.format(epoch=counter), self.gen_optimizer.get_weights())
        opt_checkpoint_path = os.path.join(self.settings.base_log_dir, "cp-opt-latent-{epoch:04d}.ckpt")
        # weights for latent opt should be zero at start of each optimizing -> so no saving
        np.save(opt_checkpoint_path.format(epoch=counter), self.latent_optimizer.get_weights())

    def load_weights(self, weight_paths: str):
        logging.info("Load weights")
        self.gen.load_weights(weight_paths)

    def load_latent_weights(self, weight_paths, optimizer_weights_path):
        self.load_weights(weight_paths)
        optimizer_weights = np.load(optimizer_weights_path, allow_pickle=True)
        # weights for latent opt should be zero at start of each optimizing -> so no loading
        self.latent_optimizer.set_weights(optimizer_weights)
