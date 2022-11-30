import argparse
import logging
import time
import datetime
import math
from pathlib import Path

import tensorflow as tf
import yaml
import sys

scene_reconstruction_folder = Path(__file__).parent.parent.parent
sys.path.append(str(scene_reconstruction_folder))
from svr.scene_reconstruction.utility.settings_manager import SettingsManager
from svr.scene_reconstruction.utility.util import limit_gpu_usage
from svr.scene_reconstruction.model.tree_model import TreeModel
from svr.scene_reconstruction.data.data_loader import DataLoader
from svr.scene_reconstruction.trainer.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start a run with the given config")
    parser.add_argument("scene_reconstruction_tf_record_folder", help="Path to the tf record folder, which contains "
                                                                      "the color, surface normals, compressed scenes "
                                                                      "and loss maps stored in tf records."
                                                                      "For help on this see data_generation/README.md.")
    parser.add_argument("log_folder", help="The folder in which the current logs are written.")
    parser.add_argument("-m", "--max_time", help="Max time this test is run in minutes", type=float, required=True)
    args = parser.parse_args()

    tf_folder = Path(args.scene_reconstruction_tf_record_folder)
    if not tf_folder.exists():
        raise FileNotFoundError("The given tf folder does not exist!")

    settings_path = Path(__file__).parent / "settings" / "settings.yaml"

    max_time_for_each_optimization = args.max_time * 60

    app_config_path = settings_path.parent / "app_config.yaml"
    folder_path = Path(args.log_folder)

    current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    folder_path = folder_path / current_time_str
    folder_path.mkdir(parents=True, exist_ok=False)

    if not settings_path.exists() or not app_config_path.exists() or not folder_path.exists():
        raise Exception(f"One of the settings files does not exist: {settings_path}, {app_config_path}, {folder_path}")

    limit_gpu_usage()

    # load settings manager
    settings_manager = SettingsManager(settings_path, app_config_path, start_logging=False)
    settings_manager.base_log_dir = folder_path
    settings_manager.app_data["DataLoader"]["tf_folder_path"] = str(tf_folder)
    settings_manager.start_logger()
    logging.info(f"Run test for {max_time_for_each_optimization}")
    logging.info(f"Base log dir: {folder_path}")

    dsl = DataLoader(settings_manager=settings_manager)
    model = TreeModel(settings_manager=settings_manager)
    trainer = Trainer(settings_manager=settings_manager, model=model, data_loader=dsl)

    time_between_train_log, time_between_val_log = 60, 60 * 5
    save_val_reach_counter = 6

    total_start_time = time.time()
    start_val_logger_time, start_train_logger_time = total_start_time, total_start_time

    epoch_counter, total_step_counter, val_log_counter = 0, 0.0, 0
    optimize_time = 0
    too_high_loss_counter = 0
    try:
        # iterate over the training dataset
        while time.time() - total_start_time < max_time_for_each_optimization:
            step_counter = 0
            for image, output, loss_factor in dsl.dataset:
                start_optimize_timer = time.time()
                trainer.optimize(image, output, loss_factor)
                if epoch_counter == 0 and total_step_counter == 0:
                    # do not take the first step, as it compiles the graph
                    total_step_counter += 1
                else:
                    optimize_time = (1.0 / total_step_counter) * (time.time() - start_optimize_timer) + (1.0 - 1.0 / total_step_counter) * optimize_time
                    total_step_counter += 1

                if time.time() - start_train_logger_time > time_between_train_log:
                    s_time = time.time()
                    # train update for tensorboard logger
                    loss, prediction, tree_prediction = trainer.get_loss(image, output, loss_factor)
                    difference = trainer.loss_manager.difference(prediction, output)
                    diff_val = tf.reduce_mean(difference).numpy()
                    with settings_manager.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=int(total_step_counter))
                        tf.summary.scalar('difference', diff_val, step=int(total_step_counter))
                    if diff_val > 2:
                        too_high_loss_counter += 1
                    else:
                        too_high_loss_counter = 0
                    if too_high_loss_counter > 10:
                        logging.info("too many high losses")
                        raise Exception("Too many high losses")
                    if math.isnan(loss.numpy()):
                        logging.info("loss is nan")
                        raise Exception("loss is nan")

                    start_train_logger_time = time.time()
                    logging.info(f"Step: {step_counter}, epoch: {epoch_counter}, step time: {optimize_time}s, took: {time.time() - s_time}s")

                if time.time() - start_val_logger_time > time_between_val_log:
                    # validation update for tensorboard logger
                    if val_log_counter + 1 == save_val_reach_counter:
                        trainer.perform_validation(int(total_step_counter), save_state=True, optimize_avg_time=optimize_time)
                        val_log_counter = 0
                    else:
                        trainer.perform_validation(int(total_step_counter), save_state=False, optimize_avg_time=optimize_time)
                        val_log_counter += 1
                    start_val_logger_time = time.time()
                step_counter += 1
                if time.time() - total_start_time > max_time_for_each_optimization:
                    logging.info(f"Max time reached: {max_time_for_each_optimization}s")
                    break
            logging.info(f"Done with epoch: {epoch_counter}, epoch_size: {step_counter}")
            # update the size of each epoch
            trainer._amount_of_steps = step_counter
            epoch_counter += 1

        logging.info("Do final validation")
        trainer.perform_validation(int(total_step_counter), save_state=True, optimize_avg_time=optimize_time,
                                   final_result_run=True)
    except Exception as e:
        logging.info(f"Exception thrown: {e}")
        result_file_path = settings_manager.base_log_dir / f"result.yaml"
        with result_file_path.open("w") as file:
            final_results = {"loss": 100000.0,
                             "optimize_avg_time": -1,
                             "node_name": trainer._node_name}
            yaml.dump(final_results, file, default_flow_style=False)
        raise e
