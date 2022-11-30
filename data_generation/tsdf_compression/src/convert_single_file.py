import logging
from typing import Union, Optional
import math

import h5py
import tensorflow as tf
import os
import numpy as np
import time
import sys
from pathlib import Path

base_dir = str(Path(__file__).parent.parent.parent / "svr" / "implicit_tsdf_decoder")
sys.path.append(base_dir)

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager
from svr.implicit_tsdf_decoder.model.dataset_loader import DataSetLoader
from svr.implicit_tsdf_decoder.model.trainer import Trainer


@tf.autograph.experimental.do_not_convert
def predict_for_file(single_file_path: Union[str, Path], output_file_path: Union[str, Path],
                     set_start_latent_vector: bool = True, database_folder: Optional[Path] = None):
    verbose = True
    selected_folder = Path(__file__).parent.parent.parent.parent / "trained_models" / "implicit_tsdf_decoder"

    if not selected_folder.exists():

        raise Exception(f"The ckpt folder does not exist: {selected_folder}")

    settings_file = Path(__file__).parent.parent.parent.parent / "svr" / "implicit_tsdf_decoder" / "settings" / "settings.yaml"
    app_config_file = settings_file.parent / "app_config.yaml"

    weights_path = os.path.join(selected_folder, "cp-0000.ckpt")

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    settings_manager = SettingsManager(str(settings_file), str(app_config_file), start_logging=False)
    settings_manager.app_data["DataLoader"]["shuffle_size"] = 0
    settings_manager.data["Training"]["batch_size"] = 64
    amount_of_latent_steps = settings_manager("Training/latent_steps")
    if set_start_latent_vector:
        amount_of_latent_steps = 750

    start_time_of_all = time.time()

    dsl = DataSetLoader(settings_manager)
    if set_start_latent_vector:
        dsl.set_start_latent_vector_use(database_folder)
    dsl.dynamically_map_classes = True
    dsl.set_paths([str(single_file_path)])
    dsl.max_amount_of_batches = 0

    dsl.init_dataset()
    dsl.dataset = dsl.dataset.cache()
    if verbose:
        logging.info(f"Batch size: {dsl.batch_size}, point amount: {dsl._point_amount}")

    org_dataset = dsl.dataset

    dsl.dataset = dsl.finalize_dataset(dsl.dataset, repeat=True)
    dataset = dsl.dataset

    trainer = Trainer(settings_manager)
    if verbose:
        print("Start caching of the dataset. To speed up the filtering.")
    start_time_of_caching = time.time()
    for e in org_dataset:
        # iterate over it once to make sure everything is cached
        continue
    if verbose:
        print(f"Took {time.time() - start_time_of_all}s to cache the dataset.")

    amount_of_inner_steps = 6
    trainer.latent_steps = amount_of_inner_steps
    trainer.latent_summary_steps = amount_of_latent_steps // amount_of_inner_steps

    # this is necessary to init the
    for points_input, output_tsdf, classes, ids, _ in dataset:
        trainer.setup_variables(points_input, output_tsdf, classes, ids)
        # latent training is necessary to create the correct optimizer variables
        trainer.init_optimize_latent_code()
        break

    trainer.load_weights(weights_path)

    batch_size = settings_manager("Training/batch_size")
    collected_latent_vars = []
    collected_batch_coords = []
    iou_results_per_batch = []
    class_results_per_batch = []
    took_time_per_batch = []

    # loop over all voxels to calculate their respective latent_value (the generator is frozen for that)
    amount_of_iterations = math.ceil(float(dsl._batch_counter) / batch_size)
    max_dsl_batch_counter = dsl._batch_counter
    start_amount_of_time = time.time()
    total_used_latent_vars = 0
    print("Finish setup -> start with iteration")
    for current_iteration in range(amount_of_iterations):
        if verbose:
            print(f"Current iteration: {current_iteration}, from: {current_iteration * batch_size} to: {(current_iteration + 1) * batch_size}, needed: {max_dsl_batch_counter}")

        # filter the datasets to select them according to their ids for the current iteration
        filtered_datasets = []
        # from id 0 to the current amount of read dsl._batch_counter
        for i in range(current_iteration * batch_size, (current_iteration + 1) * batch_size):
            # each batch in the counter must be now filtered
            if i >= dsl._batch_counter:
                break

            filtered_datasets.append(org_dataset.filter(lambda a, b, c, filter_ids, d: filter_ids[0] == i).repeat())

        choice_dataset = tf.data.Dataset.range(len(filtered_datasets)).repeat()
        sorted_dataset = tf.data.experimental.choose_from_datasets(filtered_datasets, choice_dataset)

        sorted_dataset = dsl.finalize_dataset(sorted_dataset, repeat=True, shuffle_size=0)
        # calculate for this given sorted dataset the latent values
        for points_input, output_tsdf, classes, ids, latent_vector in sorted_dataset:
            start_time = time.time()
            if set_start_latent_vector:
                trainer.setup_variables(points_input, output_tsdf, classes, ids, latent_vector)
            else:
                trainer.setup_variables(points_input, output_tsdf, classes, ids)

            for var in trainer.latent_optimizer.variables():
                var.assign(tf.zeros_like(var))
            done_latent_steps = trainer.perform_latent_opt_early_stopping()
            if verbose:
                trainer.update_current_inputs()
                results = trainer.sum_loss_value()
                info_stream = f"After {done_latent_steps * trainer.latent_summary_steps} opt steps, loss is: {trainer.loss_value().numpy():.2f}, " \
                              f"loss diff: {results['diff_total'].numpy():.5f}, loss on edge: {results['diff_on_edge'].numpy():.4f}, " \
                              f"IoU: {results['iou'].numpy()*100.0:.7f}%, IoU 90%: {results['iou_90_percent'].numpy()*100.0:.7f}%"
                iou_results_per_batch.append(results["iou"].numpy())

                if trainer.use_gradient_smoothing:
                    info_stream += f", loss gradient: {results['loss_on_gradient'].numpy()}"
                if trainer.use_classes:
                    info_stream += f", classes loss: {results['classes'].numpy():.3f}, class acc: {results['class_accuracy'] * 100.0:.3f}%"
                    class_results_per_batch.append(results["class_accuracy"].numpy())
                info_stream += f", took: {time.time() - start_time:.2f}"
                print(info_stream)
            took_time_per_batch.append(time.time() - start_time)

            # map the ids of the unique operation back to the original ids
            ordered_id_relations = {}
            for id, old_id in zip(trainer.used_ids_var.numpy().flatten(), ids.numpy().flatten()):
                if id not in ordered_id_relations:
                    ordered_id_relations[id] = old_id

            batch_coords = np.array([dsl._batch_map[ordered_id_relations[i]] for i in range(len(ordered_id_relations))])
            collected_batch_coords.append(batch_coords)

            # collect the latent values used for these filtered ids
            amount_of_used_latent_vars = len(np.unique(ids.numpy()))
            total_used_latent_vars += amount_of_used_latent_vars
            if verbose:
                print(f"Amount of used latent vars: {amount_of_used_latent_vars}")
            collected_latent_vars.append(trainer.latent_variable.numpy()[:amount_of_used_latent_vars])
            break

    collected_latent_vars = np.concatenate(collected_latent_vars, axis=0)
    collected_batch_coords = np.concatenate(collected_batch_coords, axis=0)
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_file_path), "w") as file:
        file.create_dataset(f"latent_vec", data=collected_latent_vars, compression="gzip")
        file.create_dataset("latent_locations", data=collected_batch_coords, compression="gzip")
        file.create_dataset("resolution", data=np.array([settings_manager("DataLoader/resolution")]))
        file.create_dataset("tsdf_file_path", data=np.string_(single_file_path), dtype=np.string_(single_file_path).dtype)

    if verbose:
        if trainer.use_classes:
            print(f"Took all {selected_folder.name}: {time.time() - start_time_of_all}, overall iou: {np.mean(iou_results_per_batch) * 100.0:.4f}, overall class: {np.mean(class_results_per_batch) * 100.0:.3f}")
        else:
            print(f"Took all {selected_folder.name}: {time.time() - start_time_of_all}, overall iou: {np.mean(iou_results_per_batch) * 100.0:.4f}")
    else:
        print(f"Took all {selected_folder.name}: {time.time() - start_time_of_all}")
    print(f"Took time: {time.time() - start_amount_of_time}s, for: {total_used_latent_vars}, per: {(time.time() - start_amount_of_time) / total_used_latent_vars}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Run predict single file script")
    parser.add_argument("--path", help="Path to the tsdf file", required=True)
    parser.add_argument("--output_path", help="Path to the output hdf5 file", required=True)
    args = parser.parse_args()

    predict_for_file(args.path, args.output_path)
