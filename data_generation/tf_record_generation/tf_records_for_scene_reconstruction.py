import os
import time
from typing import List, Tuple
import threading
import multiprocessing
import h5py
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_tensor_type = type(tf.constant(0))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, tf_tensor_type):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(color, normal, latent_block, average, loss_map, path):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'color': _bytes_feature(color),
        'normal': _bytes_feature(normal),
        'latent_block': _bytes_feature(latent_block),
        'average': _bytes_feature(average),
        'loss_map': _bytes_feature(loss_map),
        'path': _bytes_feature(str(path).encode('utf-8'))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(color, normal, latent_block, average, loss_map, path):
    tf_string = tf.py_function(serialize_example, (tf.io.serialize_tensor(color), tf.io.serialize_tensor(normal),
                                                   tf.io.serialize_tensor(latent_block),
                                                   tf.io.serialize_tensor(average), tf.io.serialize_tensor(loss_map),
                                                   tf.io.serialize_tensor(path)),
                               tf.string)  # the return type is `tf.string`.
    # pass these args to the above function.
    return tf.reshape(tf_string, ())


def compress_file(loss_avg_folder: Path, image_folder: Path):
    global output_folder, list_of_compressed_scenes, tf_record_files_counter
    amount_of_files_per_tf_record_file = 50
    while len(list_of_compressed_scenes) > 0:
        final_data = []
        for i in range(amount_of_files_per_tf_record_file):
            start_time = time.time()
            if len(list_of_compressed_scenes) == 0:
                break
            next_hdf5_file = Path(list_of_compressed_scenes.pop())
            loss_avg_path = loss_avg_folder / next_hdf5_file.parent.name / str(next_hdf5_file.name).replace(".hdf5", "_loss_avg.hdf5")
            color_path = image_folder / next_hdf5_file.parent.name / next_hdf5_file.name
            if not loss_avg_path.exists() or not color_path.exists():
                continue

            with h5py.File(next_hdf5_file, "r") as file:
                combined_latent_block = np.array(file["combined_latent_block"])

            path_name = next_hdf5_file.parent.name + "_" + str(next_hdf5_file.name).replace("_result.hdf5", "")

            with h5py.File(color_path, "r") as file:
                color = np.array(file["colors"], dtype=np.uint8)
                normal = np.array(file["normals"])

            if np.any(np.isnan(normal)) or np.any(np.isnan(color)):
                continue
            color = np.flip(np.transpose(color, axes=[1, 0, 2]), axis=1)
            normal = np.flip(np.transpose(normal, axes=[1, 0, 2]), axis=1)

            with h5py.File(loss_avg_path, "r") as file:
                average = np.array(file["average_16"])
                loss_map = np.array(file["lossmap_valued"]).astype(np.uint8)

            final_data.append((color, normal, combined_latent_block, average, loss_map, path_name))
            print("Done for {}, took: {:.3f}".format(i, time.time() - start_time))

        if len(final_data) > 0:
            start_time = time.time()
            tf_record_file_path = os.path.join(output_folder, f"{tf_record_files_counter}.tfrecord")
            tf_record_files_counter += 1

            def generator():
                for color, normal, combined_latent_block, average, loss_map, path_name in final_data:
                    yield color, normal, combined_latent_block, average, loss_map, path_name
            output_shape = ((512, 512, 3), (512, 512, 3), (16, 16, 16, 512), (16, 16, 16), (256, 256, 256), None)
            output_type = (tf.uint8, tf.float32, tf.float32, tf.float32, tf.uint8, tf.string)
            dataset = tf.data.Dataset.from_generator(generator, output_shapes=output_shape, output_types=output_type)

            dataset = dataset.map(tf_serialize_example)
            writer = tf.data.experimental.TFRecordWriter(tf_record_file_path, compression_type="GZIP")
            writer.write(dataset)
            print("Done writing took: {:.3f}".format(time.time() - start_time))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Convert the created scene into tf record files for training!")
    parser.add_argument("compressed_scene_folder", help="Folder in which the compressed latent blocks are stored!")
    parser.add_argument("image_folder", help="Folder in which the images are stored")
    parser.add_argument("tsdf_point_cloud_folder", help="Folder in which the tsdf point cloud and the loss maps are stored.")
    parser.add_argument("tf_record_output_folder", help="Folder in which the tf records are stored in the end")
    args = parser.parse_args()

    output_folder = Path(args.tf_record_output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    compressed_scene_folder = Path(args.compressed_scene_folder)
    if not compressed_scene_folder.exists():
        raise FileNotFoundError("The compressed scene folder does not exist!")

    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError("The image folder does not exist!")

    tsdf_point_folder = Path(args.tsdf_point_cloud_folder)
    if not tsdf_point_folder.exists():
        raise FileNotFoundError("The tsdf point cloud folder does not exist!")

    list_of_compressed_scenes = list(compressed_scene_folder.glob("**/*.hdf5"))

    print(f"Found {len(list_of_compressed_scenes)} hdf5 files")
    tf_record_files_counter = 0


    amount_of_threads = multiprocessing.cpu_count()
    threads = []
    for i in range(amount_of_threads):
        thread = threading.Thread(target=compress_file, args=(tsdf_point_folder, image_folder))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        time.sleep(1)

    for thread in threads:
        thread.join()

