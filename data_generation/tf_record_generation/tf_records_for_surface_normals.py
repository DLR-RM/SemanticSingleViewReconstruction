from pathlib import Path
import argparse
import os
from typing import List, Tuple
import multiprocessing
import threading

import h5py
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_tensor_type = type(tf.constant(0))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, tf_tensor_type):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(color, normal):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'color': _bytes_feature(color),
        'normal': _bytes_feature(normal),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(color, normal):
    tf_string = tf.py_function(serialize_example, (tf.io.serialize_tensor(color), tf.io.serialize_tensor(normal)),
                               tf.string)  # the return type is `tf.string`.
    # pass these args to the above function.
    return tf.reshape(tf_string, ())


def write_data_list_to_tf_record(final_data: List[Tuple[np.ndarray, np.ndarray]]):
    global output_folder, tf_counters

    def generator():
        for color, normal in final_data:
            yield color, normal

    output_shape = ((512, 512, 3), (512, 512, 3))
    output_type = (tf.float32, tf.float32)
    dataset = tf.data.Dataset.from_generator(generator, output_shapes=output_shape, output_types=output_type)

    # remove elements if they are non
    dataset = dataset.filter(lambda color, normal:
                             not tf.math.reduce_any(tf.math.is_nan(color)) and
                             not tf.math.reduce_any(tf.math.is_nan(normal)))
    # map color image to weird output space
    dataset = dataset.map(lambda color, normal:
                          (tf.reverse(tf.transpose(color, perm=(1, 0, 2)), axis=[1]),
                           tf.reverse(tf.transpose(normal, perm=(1, 0, 2)), axis=[1])),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(tf_serialize_example)

    tf_record_file_path = output_folder / f"{len(tf_counters)}.tfrecord"
    tf_counters.append(0)

    output_folder.mkdir(parents=True, exist_ok=True)
    writer = tf.data.experimental.TFRecordWriter(str(tf_record_file_path), compression_type="GZIP")
    writer.write(dataset)


def compress_file():
    global image_files
    current_data_list = []
    while len(image_files) > 0:
        image_file = image_files.pop()
        with h5py.File(image_file, "r") as file:
            if "colors" not in file.keys():
                continue
            color = np.array(file["colors"], dtype=np.float32) / 255.0
            if "normals" in file.keys():
                normal = np.array(file["normals"])
            else:
                # texture swapped, find correct normal image
                amount_of_camposes = int(sum([1 for new_path in image_file.parent.iterdir() if new_path.is_file()]) / 4)
                current_file_nr = int(image_file.with_suffix("").name)
                used_file_nr = current_file_nr % amount_of_camposes
                with h5py.File(image_file.parent / f"{used_file_nr}.hdf5") as file3:
                    if "normals" in file3.keys():
                        normal = np.array(file3["normals"])
                    else:
                        continue
            current_data_list.append((color, normal))
            if len(current_data_list) < 100:
                continue
            write_data_list_to_tf_record(current_data_list)
            current_data_list = []
    if len(current_data_list) > 0:
        write_data_list_to_tf_record(current_data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create a tf record for all color and surface normal images")
    parser.add_argument("image_folder", help="Path to the image folder")
    parser.add_argument("output_folder", help="Path where the tf record files will be stored")
    args = parser.parse_args()

    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError("The image folder does not exist!")

    image_files = list(image_folder.glob("**/*.hdf5"))
    output_folder = Path(args.output_folder)
    tf_counters = []

    print(f"Found: {len(image_files)}")

    use_multithreading = True
    if not use_multithreading:
        compress_file()
    else:
        amount_of_threads = multiprocessing.cpu_count()

        threads = []
        for i in range(amount_of_threads):
            thread = threading.Thread(target=compress_file)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
