import math
import glob
from multiprocessing import Process, Manager, Value
import multiprocessing
from pathlib import Path

import time
import os
import argparse
# turn of the graphics card
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager
from svr.implicit_tsdf_decoder.model.dataset_loader import DataSetLoader


def start_thread(thread_id, batch_counter, path_counter, mutex, used_folder_path, store_folder_path, max_value):
    settings_folder = Path(__file__).parent.parent.parent / "svr" / "implicit_tsdf_decoder" / "settings"
    settings = SettingsManager(settings_folder / "settings.yaml", settings_folder / "app_config.yaml",
                               start_logging=False)
    # set the folder path a new with the given variable
    settings.set_element("DataLoader/folder_path", used_folder_path)
    settings.set_element("DataLoader/tf_folder_path", store_folder_path)

    batch_size = settings('Training/batch_size')
    point_amount = settings('Training/point_amount')
    resolution = settings("DataLoader/resolution")
    print(f"Current batch size: {batch_size}, point amount: {point_amount}")

    dataset_loader = DataSetLoader(settings)
    # to change the classes on the file, we have to change this here to true
    dataset_loader.dynamically_map_classes = True
    dataset_loader.set_shared_batch_counter(batch_counter, mutex)

    paths = glob.glob(os.path.join(dataset_loader.folder_path, "*", "*.hdf5"))
    paths = [path for path in paths if "_loss_avg.hdf5" not in path]
    paths.sort()
    paths = paths[:max_value]
    amount_of_paths = len(paths)
    number_per_tf_file = 50
    amount_of_final_tf_files = math.ceil(amount_of_paths / float(number_per_tf_file))
    path_max_len = np.max([len(p) for p in paths]) + 1
    mutex.acquire()
    tf_folder_path = settings("DataLoader/tf_folder_path")
    tf_record_path = tf_folder_path
    if not os.path.exists(tf_record_path):
        os.makedirs(tf_record_path)
    mutex.release()
    print(f"There will be {amount_of_final_tf_files} final tf files.")
    org_paths = paths
    while True:
        mutex.acquire()
        used_path_counter = path_counter.value
        path_counter.value += 1
        mutex.release()

        start_id = int(amount_of_paths * used_path_counter / float(amount_of_final_tf_files))

        end_id = int(amount_of_paths * (1+used_path_counter) / float(amount_of_final_tf_files))
        paths = org_paths[start_id:end_id]
        print(f"Start: {start_id}, end: {end_id} for {used_path_counter}")
        print("Use {} paths, max len is {}".format(len(paths), path_max_len))
        if len(paths) == 0:
            break
        dataset_loader.set_paths(paths)
        dataset_loader.init_dataset()

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value.numpy()))

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def serialize_example(points, dists, classes, batch_counter):
            """
            Creates a tf.train.Example message ready to be written to a file.
            """
            # Create a dictionary mapping the feature name to the tf.train.Example-compatible
            # data type.
            feature = {
                'points': _bytes_feature(points),
                'dists': _bytes_feature(dists),
                'classes': _bytes_feature(classes),
                'batch_counter': _int64_feature(batch_counter)
            }

            # Create a Features message using tf.train.Example.

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        def tf_serialize_example(points, dists, classes, batch_counter, sim_vec):
            tf_string = tf.py_function(serialize_example, (tf.io.serialize_tensor(points), tf.io.serialize_tensor(dists), tf.io.serialize_tensor(classes), batch_counter),
                                       tf.string)  # the return type is `tf.string`.
            # pass these args to the above function.
            return tf.reshape(tf_string, ())

        tf_record_file_path = os.path.join(tf_record_path, f"{used_path_counter}.tfrecord")
        dataset = dataset_loader.dataset.map(tf_serialize_example)
        writer = tf.data.experimental.TFRecordWriter(tf_record_file_path, compression_type="GZIP")
        writer.write(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate tf record data for the training of the implicit tsdf compression, "
                                     "for the given folder path")
    parser.add_argument("input_folder", help="Path to the folder in which the */*.hdf5 containers are stored.")
    parser.add_argument("output_folder", help="Path in which the resulting folder will be saved.")
    parser.add_argument("--max_value", help="Maximum amount of files used", default=1000, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise Exception(f"This path does not exist: {args.input_folder}")

    start_time = time.time()
    processes = []
    manager = Manager()
    batch_counter = Value("i", 0)
    path_counter = Value("i", 0)
    folder_path = os.path.abspath(args.input_folder)
    output_folder_path = os.path.abspath(args.output_folder)
    mutex = manager.Lock()

    amount_of_threads = 1 #multiprocessing.cpu_count()
    for i in range(amount_of_threads):
        processes.append(Process(target=start_thread, args=(i, batch_counter, path_counter, mutex, folder_path, output_folder_path, args.max_value)))
        processes[-1].start()

    for p in processes:
        p.join()

    print(time.time()-start_time)
