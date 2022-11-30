import socket
import time
import os
import h5py
import logging

import yaml
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

from svr.scene_reconstruction.data.data_loader import DataLoader
from svr.scene_reconstruction.trainer.loss_manager import LossManager
from svr.scene_reconstruction.utility.settings_manager import SettingsManager


class Trainer(object):

    def __init__(self, settings_manager: SettingsManager, model: Model, data_loader: DataLoader):
        self.settings = settings_manager
        self.model = model
        self.data_loader = data_loader
        self._dataset = data_loader.dataset
        self._val_dataset = data_loader.validation_dataset
        self.batch_size = data_loader.batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=settings_manager("Training/learning_rate"))
        self.loss_manager = LossManager(settings_manager)
        self._amount_of_steps = 0
        self.start_time = time.time()
        self._node_name = socket.gethostname()

    @tf.function
    def optimize(self, image, output, loss_factor):
        with tf.GradientTape() as tape:
            prediction, tree_prediction = self.model(image, training=True)
            loss = self.loss_manager.loss(prediction, tree_prediction, output, loss_factor)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def get_loss(self, image, output, loss_factor):
        prediction, tree_prediction = self.model(image, training=True)
        loss = self.loss_manager.loss(prediction, tree_prediction, output, loss_factor)
        return loss, prediction, tree_prediction

    @tf.function
    def combine_all_losses(self, image, output, loss_factor, reachable_area_factor=None):
        loss, prediction, tree_prediction = self.get_loss(image, output, loss_factor)
        result = {"scalar": {}, "hist": {}}

        def reduce(x, **kwargs):
            return tf.reduce_mean(x, **kwargs)
        difference = self.loss_manager.difference(prediction, output)
        tree_difference = self.loss_manager.difference(tree_prediction, output)

        # loss
        result["scalar"]["loss"] = reduce(loss)
        result["scalar"]["single_loss"] = reduce(self.loss_manager.loss_single(prediction, output, loss_factor))
        result["scalar"]["tree/loss"] = reduce(self.loss_manager.loss_single(tree_prediction, output, loss_factor))

        result["hist"]["single_loss"] = reduce(self.loss_manager.loss_single(prediction, output, loss_factor), axis=-1)
        result["hist"]["tree/loss"] = reduce(self.loss_manager.loss_single(tree_prediction, output, loss_factor), axis=-1)

        # differences
        result["scalar"]["difference"] = reduce(difference)
        result["scalar"]["tree/difference"] = reduce(tree_difference)
        result["hist"]["difference"] = reduce(difference, axis=-1)
        result["hist"]["tree/difference"] = reduce(tree_difference, axis=-1)

        if reachable_area_factor is not None:
            # differences
            reachable_area_factor = reachable_area_factor
            result["scalar"]["difference_reachable"] = reduce(tf.reduce_mean(difference, axis=-1) * reachable_area_factor)
            result["scalar"]["tree/difference_reachable"] = reduce(tf.reduce_mean(tree_difference, axis=-1) * reachable_area_factor)
            result["hist"]["difference_reachable"] = reduce(difference, axis=-1) * reachable_area_factor
            result["hist"]["tree/difference_reachable"] = reduce(tree_difference, axis=-1) * reachable_area_factor

        # output and prediction
        result["hist"]["output"] = output
        result["hist"]["prediction"] = prediction
        result["hist"]["tree/prediction"] = tree_prediction

        return result

    @staticmethod
    def combine_results_of_all_losses(all_results):
        # convert the long list into a dict with many single lists
        combined_results = {}
        for single_result in all_results:
            for key, selected_results in single_result.items():
                if key not in combined_results:
                    combined_results[key] = {}
                for loss_name, data_point in selected_results.items():
                    if loss_name not in combined_results[key]:
                        combined_results[key][loss_name] = [data_point]
                    else:
                        combined_results[key][loss_name].append(data_point)
        return combined_results

    @staticmethod
    def save_single_example_to_tensorboard(single_result):
        # save all new elements into the tf record file
        for key, selected_results in single_result.items():
            for loss_name, data_point in selected_results.items():
                if key == "scalar":
                    tf.summary.scalar(loss_name, data_point)
                elif key == "hist":
                    used_loss_name = loss_name
                    if "tree" in used_loss_name:
                        used_loss_name = used_loss_name.replace("tree/", "tree/hist_")
                    else:
                        used_loss_name = "hist_" + used_loss_name
                    tf.summary.histogram(used_loss_name, data_point)
                else:
                    raise Exception(f"This type is unknown: {key}")

    @staticmethod
    def save_to_tensorboard(combined_results):
        # save all new elements into the tf record file
        for key, selected_results in combined_results.items():
            if key == "scalar":
                for loss_name, current_list in selected_results.items():
                    tf.summary.scalar(loss_name, float(np.mean(current_list)))
            elif key == "hist":
                for loss_name, current_list in selected_results.items():
                    used_loss_name = loss_name
                    if "tree" in used_loss_name:
                        used_loss_name = used_loss_name.replace("tree/", "tree/hist_")
                    else:
                        used_loss_name = "hist_" + used_loss_name
                    tf.summary.histogram(used_loss_name, np.mean(current_list, axis=0))
            else:
                raise Exception(f"This type of summary is unknown: {key}")


    def perform_validation(self, current_step_count: int, save_state: bool, optimize_avg_time: float,
                           final_result_run: bool = False):
        start_time = time.time()
        # calculate the loss for each validation data point
        all_results = []

        for index, (image, output, loss_factor, reachable_area) in enumerate(self._val_dataset):
            all_results.append(self.combine_all_losses(image, output, loss_factor, reachable_area))
            # only check the first two hundred images -> that should be enough
            if index > 200 // self.batch_size:
                break

        with self.settings.val_summary_writer.as_default(step=current_step_count):
            combined_results = self.combine_results_of_all_losses(all_results)
            self.save_to_tensorboard(combined_results)
        self.settings.val_summary_writer.flush()
        mean_loss = np.mean(combined_results['scalar']['loss'])
        logging.info(f"Loss: {mean_loss}, diff: {np.mean(combined_results['scalar']['difference'])}, validation: {time.time() - start_time:.2f}s")
        if final_result_run:
            result_file_path = self.settings.base_log_dir / f"result.yaml"
        else:
            result_file_path = self.settings.base_log_dir / f"result_{current_step_count}.yaml"
        with result_file_path.open("w") as file:
            final_results = {"loss": np.mean(mean_loss),
                             "optimize_avg_time": optimize_avg_time,
                             "node_name": self._node_name}
            yaml.dump(final_results, file, default_flow_style=False)

        if save_state:
            self.save_model(current_step_count)
            self.save_output(current_step_count)

    def create_graph_and_profile(self):
        for image, output, loss_factor in self._dataset:
            tf.summary.trace_on(graph=True, profiler=True)
            self.optimize(image, output, loss_factor)
            with self.settings.train_summary_writer.as_default():
                tf.summary.trace_export(name="optimize", step=0, profiler_outdir=self.settings.train_log_dir)
            break
        logging.info("Create graph and profile, programm should stop")
        exit(0)

    def save_model(self, epoch: int):
        logging.info(f"Save the model")
        checkpoint_path = os.path.join(self.settings.base_log_dir, "cp-{epoch:08d}.ckpt")
        self.model.save_weights(checkpoint_path.format(epoch=epoch))
        opt_checkpoint_path = os.path.join(self.settings.base_log_dir, "cp-opt-model-{epoch:08d}.ckpt")
        # weights for latent opt should be zero at start of each optimizing -> so no saving
        np.save(opt_checkpoint_path.format(epoch=epoch), self.model.get_weights())

    def save_output(self, epoch: int):
        logging.info(f"Save the output")
        for image, output, loss_factor, reachable_area in self._val_dataset:
            prediction, _ = self.model(image, training=True)
            with h5py.File(
                    os.path.join(self.settings.base_log_dir, "predicted_output_{epoch:08d}.hdf5").format(epoch=epoch),
                    "w") as file:
                prediction = prediction.numpy()
                output = output.numpy()
                image = image.numpy()
                for batch_nr in range(np.min([prediction.shape[0], 3])):
                    file.create_dataset(f"prediction_{batch_nr}", data=prediction[batch_nr], compression="gzip")
                    file.create_dataset(f"output_{batch_nr}", data=output[batch_nr], compression="gzip")
                    file.create_dataset(f"image_{batch_nr}", data=image[batch_nr], compression="gzip")
            break
