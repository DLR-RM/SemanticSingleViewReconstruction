import socket
import time
import os
import logging

import yaml
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

from svr.u_net_normal.normal_reconstruction.data.data_loader import DataLoader
from svr.u_net_normal.normal_reconstruction.trainer.loss_manager import LossManager
from svr.u_net_normal.normal_reconstruction.utility.settings_manager import SettingsManager


class Trainer(object):

    def __init__(self, settings_manager: SettingsManager, model: Model, data_loader: DataLoader):
        self.settings = settings_manager
        self.model = model
        self.data_loader = data_loader
        self._dataset = data_loader.dataset
        self._val_dataset = data_loader.validation_dataset
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=settings_manager("Training/learning_rate"))
        self.loss_manager = LossManager(settings_manager)
        self.batch_size = self.settings("Training/batch_size")
        self._amount_of_steps = 0
        self.start_time = time.time()
        self._node_name = socket.gethostname()
        self.mirrored_strategy = None

    @tf.function
    def optimize(self, image, output):
        with tf.GradientTape() as tape:
            prediction = self.model(image, training=True)
            loss = self.loss_manager.loss(prediction, output)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def get_loss(self, image, output):
        prediction = self.model(image, training=True)
        loss = self.loss_manager.loss(prediction, output)
        return loss, prediction


    def combine_all_losses(self, image, output, add_image: bool = False):
        loss, prediction = self.get_loss(image, output)
        result = {"scalar": {}, "hist": {}, "image": {}}

        def reduce(x, **kwargs):
            return tf.reduce_mean(x, **kwargs).numpy()

        # loss
        result["scalar"]["loss"] = reduce(loss)
        result["scalar"]["cosine_distance"] = reduce(self.loss_manager.cosine_loss(prediction, output))

        angels_diffs = self.loss_manager.angels_percentage(prediction, output)
        result["scalar"].update(angels_diffs)

        result["hist"]["cosine_distance"] = self.loss_manager.cosine_loss(prediction, output).numpy()
        angels_diff = self.loss_manager.angels_diff_image(prediction, output)
        result["hist"]["angels_diff"] = angels_diff

        # output and prediction
        result["hist"]["output"] = output.numpy()
        result["hist"]["prediction"] = prediction.numpy()

        if add_image:
            result["image"]["output"] = self.data_loader.convert_image_back(output[0], mode="normal")
            result["image"]["prediction"] = self.data_loader.convert_image_back(prediction[0], mode="normal")
            result["image"]["input"] = self.data_loader.convert_image_back(image[0], mode="color")
            result["image"]["angels_diff"] = self.data_loader.convert_image_back(angels_diff[0] / 60.0, mode="")
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
    def save_to_tensorboard(combined_results):
        # save all new elements into the tf record file
        for key, selected_results in combined_results.items():
            if key == "scalar":
                for loss_name, current_list in selected_results.items():
                    tf.summary.scalar(loss_name, float(np.mean(current_list)))
            elif key == "hist":
                for loss_name, current_list in selected_results.items():
                    used_loss_name = "hist_" + loss_name
                    tf.summary.histogram(used_loss_name, np.mean(current_list, axis=0))
            elif key == "image":
                for loss_name, current_list in selected_results.items():
                    used_loss_name = "image/" + loss_name
                    if len(current_list) > 0:
                        current_list = np.array(current_list)[:1]
                        if len(current_list.shape) == 3:
                            current_list = current_list.reshape((current_list.shape[0], current_list.shape[1], current_list.shape[2], 1))
                        tf.summary.image(used_loss_name, current_list)
            else:
                raise Exception(f"This type of summary is unknown: {key}")

    def perform_validation(self, current_step_count: int, save_state: bool, optimize_avg_time: float,
                           final_result_run: bool = False):
        start_time = time.time()
        # calculate the loss for each validation data point
        all_results = [self.combine_all_losses(image, output, add_image=True) for image, output in self._val_dataset]

        with self.settings.val_summary_writer.as_default(step=current_step_count):
            combined_results = self.combine_results_of_all_losses(all_results)
            self.save_to_tensorboard(combined_results)
        self.settings.val_summary_writer.flush()
        mean_loss = np.mean(combined_results['scalar']['loss'])
        logging.info(f"Loss: {mean_loss}, validation: {time.time() - start_time:.2f}s")
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

