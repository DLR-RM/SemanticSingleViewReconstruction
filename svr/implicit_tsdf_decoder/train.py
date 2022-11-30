import os
from pathlib import Path
import shutil
import argparse
import logging
import time
import datetime

import yaml
import tensorflow as tf
import sys

svr_folder = Path(__file__).parent.parent.parent
sys.path.append(str(svr_folder))

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager
from svr.implicit_tsdf_decoder.model.trainer import Trainer
from svr.implicit_tsdf_decoder.model.dataset_loader import DataSetLoaderTraining
from svr.implicit_tsdf_decoder.utility.util import limit_gpu_usage, save_current_state_to_folder, \
    create_summary_writers, perform_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start a training run")
    parser.add_argument("tsdf_point_cloud_tf_records_folder", help="Path to the tf record folder, which contains the "
                                                                   "tsdf point clouds. For help on this see "
                                                                   "data_generation/README.md.")
    parser.add_argument("log_folder", help="The folder in which the current logs are written.")
    parser.add_argument("-m", "--max_time", help="Max time this test is run in minutes", type=float, required=True)
    args = parser.parse_args()

    tf_folder = Path(args.tsdf_point_cloud_tf_records_folder)
    if not tf_folder.exists():
        raise FileNotFoundError("The given tf record folder does not exist!")

    max_time_for_each_optimization = args.max_time * 60
    settings_path = Path(__file__).parent / "settings" / "settings.yaml"
    app_config_path = settings_path.parent / "app_config.yaml"

    folder_path = Path(args.log_folder)
    current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    folder_path = folder_path / current_time_str
    folder_path.mkdir(parents=True, exist_ok=False)

    if not os.path.exists(settings_path) or not os.path.exists(app_config_path) or not os.path.exists(folder_path):
        raise Exception(f"One of the settings files does not exist: {settings_path}, {app_config_path}, {folder_path}")

    limit_gpu_usage()

    # load settings manager
    settings_manager = SettingsManager(settings_path, app_config_path, start_logging=False)
    settings_manager.base_log_dir = folder_path
    settings_manager.app_data["DataLoader"]["tf_folder_path"] = str(tf_folder)
    settings_manager.start_logger()
    logging.info(f"Run test for {max_time_for_each_optimization}")
    # save the current status
    save_current_state_to_folder(settings_manager)

    # save the current used settings
    folder_path = Path(settings_manager.base_log_dir)
    new_settings_path = folder_path / "used_settings.yaml"
    shutil.copy2(settings_path, new_settings_path)

    dsl = DataSetLoaderTraining(settings_manager)
    dataset = dsl.dataset

    trainer = Trainer(settings_manager)

    train_summary_writer, latent_summary_writer, validation_summary_writer, \
        validation_mean_summary_writer = create_summary_writers(settings_manager)

    latent_steps = settings_manager("Training/latent_steps")
    gen_steps = settings_manager("Training/gen_steps")
    summary_steps = settings_manager("Training/summary_steps")

    counter = 0
    validation_counter = 0

    trainer_loss_max_value = 500.0
    logging.info("Start timer")
    start_time = time.time()
    for points_input, output_tsdf, classes, ids in dataset:
        # setup all variables
        trainer.setup_variables(points_input, output_tsdf, classes, ids)

        with latent_summary_writer.as_default():
            tf.summary.histogram('used_ids_var', trainer.used_ids_var, step=counter)
            tf.summary.histogram('current_ids', ids, step=counter)

        def latent_update(n_trainer, step):
            with latent_summary_writer.as_default():
                tf.summary.scalar('loss_it', n_trainer.loss_value(), step=step * summary_steps + latent_steps * counter)
        trainer.perform_latent_opt(latent_update)

        trainer.add_loss_summaries_to_writer(latent_summary_writer, counter)
        if counter == 0:
            trainer.gen.summary()

        def gen_update(n_trainer, step):
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_it', n_trainer.loss_value(), step=step * summary_steps + gen_steps * counter)
        trainer.perform_gen_training(gen_update)
        trainer.add_loss_summaries_to_writer(train_summary_writer, counter)
        if trainer.loss_value().numpy() > trainer_loss_max_value:
            logging.info(f"The trainer loss value is far too high, above {trainer_loss_max_value}: {trainer.loss_value().numpy()}, stop training.")
            break

        logging.info("Done with step: {}".format(counter))
        counter += 1

        if time.time() - start_time > max_time_for_each_optimization:
            print("Maximum time reached")
            break
        elif time.time() - start_time > max_time_for_each_optimization * 0.5:
            logging.info("Do half time validation")
            validation_counter, _, _ = perform_validation(trainer, dsl, validation_summary_writer, validation_mean_summary_writer, validation_counter)

    logging.info("Do final validation")
    validation_counter, collection_of_losses, time_measurements = perform_validation(trainer, dsl, validation_summary_writer, validation_mean_summary_writer, validation_counter)
    final_results = {}
    final_results["time_measurements_latent_validation"] = time_measurements
    for key, data in collection_of_losses.items():
        final_results[key] = tf.reduce_mean(data).numpy()

    final_path_result = os.path.join(folder_path, "result.yaml")
    # save as yaml
    with open(final_path_result, "w") as file:
        yaml.dump(final_results, file, default_flow_style=False)

    trainer.save_weights(0)
