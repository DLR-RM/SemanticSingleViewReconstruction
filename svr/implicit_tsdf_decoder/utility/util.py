import glob
import shutil
import os
import logging
import time
from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager, AppConfig, ModelParams
from svr.implicit_tsdf_decoder.model.trainer import Trainer


def save_current_state_to_folder(settings_manager: SettingsManager):
    """
    Saves the full source folder including py, yml and yaml files to the log
    To make results better reproducible

    :param settings_manager:
    """
    folder_path = settings_manager.base_log_dir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    src_folder = Path(__file__).parent.parent
    paths = []
    for extension in ["py", "yaml", "yml"]:
        paths.extend(glob.glob(os.path.join(src_folder, "**", f"*.{extension}")))
    goal_folder = os.path.join(folder_path, "used_src")
    for path in paths:
        file_name = os.path.abspath(path)[len(str(src_folder.absolute())):].strip("/")
        goal_path = os.path.join(goal_folder, file_name)
        if not os.path.exists(os.path.dirname(goal_path)):
            os.makedirs(os.path.dirname(goal_path))
        shutil.copy2(path, goal_path)


def create_summary_writers(settings_manager: SettingsManager):
    train_summary_writer = tf.summary.create_file_writer(settings_manager.train_log_dir)
    latent_summary_writer = tf.summary.create_file_writer(settings_manager.latent_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(settings_manager.validation_log_dir)
    validation_mean_summary_writer = tf.summary.create_file_writer(settings_manager.validation_mean_log_dir)

    model_params = ModelParams(settings_manager)
    app_config = AppConfig(settings_manager)

    metric_validation_mean_loss = hp.Metric("loss", display_name="loss")
    metric_validation_mean_loss_diff = hp.Metric("loss_diff", display_name="loss_diff")
    metric_validation_mean_loss_diff_on_edge = hp.Metric("loss_diff_on_edge", display_name="loss_diff_on_edge")
    metric_validation_mean_iou = hp.Metric("mean_iou", display_name="mean_iou")
    metrics = [metric_validation_mean_loss, metric_validation_mean_loss_diff,
               metric_validation_mean_loss_diff_on_edge, metric_validation_mean_iou]

    with validation_mean_summary_writer.as_default():
        hp.hparams_config(model_params.hparams, metrics=metrics)
        hp.hparams_config(app_config.hparams, metrics=metrics)
        hp.hparams(model_params.get_hparams())
        hp.hparams(app_config.get_hparams())
    return train_summary_writer, latent_summary_writer, validation_summary_writer, validation_mean_summary_writer

def limit_gpu_usage():
    gpus = tf.config.list_physical_devices('GPU')
    # Currently, memory growth needs to be the same across GPUs
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def perform_validation(trainer: Trainer, dsl, validation_summary_writer,
                       validation_mean_summary_writer, validation_counter,
                       amount_of_validation_checks=None):
    start_time_of_val = time.time()
    if amount_of_validation_checks is None:
        amount_of_validation_checks = trainer.settings("Training/amount_of_validation_checks")
    validation_counter_inside = 0

    latent_steps = trainer.settings("Training/latent_steps")
    summary_steps = trainer.settings("Training/summary_steps")

    collection_of_losses = {}
    time_measurements = []
    for v_points_input, v_output_tsdf, v_classes, v_ids in dsl.validation_dataset:
        # setup all variables
        trainer.setup_variables(v_points_input, v_output_tsdf, v_classes, v_ids)
        with validation_summary_writer.as_default():
            tf.summary.histogram('used_ids_var', trainer.used_ids_var, step=validation_counter)
            tf.summary.histogram('current_ids', v_ids, step=validation_counter)

        def val_latent_update(n_trainer, step):
            with validation_summary_writer.as_default():
                tf.summary.scalar('loss_it', n_trainer.loss_value(),
                                  step=step * summary_steps + latent_steps * validation_counter)

        start_time = time.time()
        trainer.perform_latent_opt(val_latent_update)
        time_measurements.append(time.time() - start_time)
        results = trainer.sum_loss_value()
        collection_of_losses.setdefault("loss", []).append(trainer.loss_value())
        collection_of_losses.setdefault("loss_diff", []).append(results["diff_total"])
        collection_of_losses.setdefault('loss_diff_on_edge', []).append(results["diff_on_edge"])
        collection_of_losses.setdefault('mean_iou', []).append(results["iou"])
        if trainer.use_gradient_smoothing:
            collection_of_losses.setdefault('loss_gradient_smoothing', []).append(results["loss_on_gradient"])
        if trainer.use_classes:
            collection_of_losses.setdefault('classes', []).append(results["classes"])
            collection_of_losses.setdefault('scaled_classes', []).append(results["classes"] * trainer.loss_manager._class_weighting_loss)
            collection_of_losses.setdefault('class_accuracy', []).append(results["class_accuracy"])
        trainer.add_loss_summaries_to_writer(validation_summary_writer, validation_counter)
        if amount_of_validation_checks == validation_counter_inside:
            break
        validation_counter_inside += 1
        validation_counter += 1
    logging.info(f"Validation took: {time.time() - start_time_of_val}")

    with validation_mean_summary_writer.as_default():
        for key, data in collection_of_losses.items():
            tf.summary.scalar(key, tf.reduce_mean(data), step=validation_counter)
    return validation_counter, collection_of_losses, time_measurements