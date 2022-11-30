
from pathlib import Path
import warnings
import glob
import os

import tensorflow as tf
import h5py
import numpy as np

from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager
from svr.implicit_tsdf_decoder.model.filter_classes import FilterClasses
from svr.implicit_tsdf_decoder.model.similarity_setter import SimilaritySetter
from svr.implicit_tsdf_decoder.model.convert_tsdf_to_blocked_tsdf_file import convert_tsdf_to_blocked_tsdf_file_path, \
    convert_blocked_to_similarities


class DataSetLoader(object):

    def __init__(self, settings_manager: SettingsManager):
        self.settings = settings_manager
        self._resolution = self.settings("DataLoader/resolution")
        self._point_amount = self.settings("Training/point_amount")
        self._latent_dim = self.settings("Generator/latent_dim")
        self._min_point_amount_for_block = int(self._point_amount * self.settings("DataLoader/min_point_amount"))
        self._select_one_batch_per_cube = self.settings("DataLoader/select_one_batch_per_cube")
        self._shuffle_size = self.settings("DataLoader/shuffle_size")
        self.batch_size = self.settings("Training/batch_size")
        self.mode = self.settings("DataLoader/mode")
        self.load_only_blocks_with_boundary = self.settings("DataLoader/load_only_blocks_with_boundary")
        self.amount_of_blocks_per_voxel = self.settings("DataLoader/amount_of_blocks_per_voxel")
        self._org_truncation_threshold = self.settings("DataLoader/org_trunc_threshold")  # fixed in the DataSetLoader
        if self.mode == "normal":
            self._truncation_threshold = self._org_truncation_threshold / (2.0 / self._resolution)
        else:
            self._truncation_threshold = 1.0
        self._truncation_boundary_selection_value = self._truncation_threshold * self.settings("DataLoader/tsdf_min_threshold_to_use_block")
        # add the correct trunc threshold to the settings
        self.settings.data["Generator"]["trunc_threshold"] = self._truncation_threshold
        self.max_amount_of_batches = 0
        self.dataset = None
        self.paths = []
        self._batch_counter = 0
        self._shared_batch_counter = None
        self._shared_batch_mutex = None
        self.dynamically_map_classes = False
        self.boundary_selection_scale = self.settings("DataLoader/boundary_selection_scale")
        self.boundary_selection_scale_half = 0
        if self.boundary_selection_scale != 0:
            self.boundary_selection_scale_half = (self.boundary_selection_scale - 1.0) * 0.5 + 1.0
        self._filter_classes = FilterClasses()
        self._batch_map = {}
        self._amount_of_points_list = []
        self._return_start_latent_vector = False
        self._start_latent_vector = np.zeros(self._latent_dim)
        self._only_non_boundary_mode = False

    def set_start_latent_vector_use(self, database_folder: Path):
        self._return_start_latent_vector = True
        if self._return_start_latent_vector:
            self.sim_setter = SimilaritySetter(database_folder, only_non_boundary=self._only_non_boundary_mode)

    def set_only_non_boundary_mode(self):
        self._only_non_boundary_mode = True

    def set_shared_batch_counter(self, batch_counter, mutex):
        self._shared_batch_counter = batch_counter
        self._shared_batch_mutex = mutex

    def get_batch_counter(self):
        return self._batch_counter

    def get_extract_batches(self, path, given_similarity_file_path=None, only_non_boundary: bool = False):
        if not os.path.exists(path):
            raise Exception(f"Path not found: {path}")
        if self.max_amount_of_batches > 0:
            warnings.warn(f"Max amount of batches is used: {self.max_amount_of_batches}")
        if not self._return_start_latent_vector:
            use_blocker_to_create_blocker_file = False
            with h5py.File(path, "r") as file:
                if "points_dist" in file.keys():
                    point_dist = np.array(file["points_dist"])
                    points = point_dist[:, :, :3].copy()
                    dists = point_dist[:, :, 3].copy()
                    class_batch_counter = np.array(file["class_batch_counter"])
                    classes = class_batch_counter[:, :, 0]
                    batch_counters = class_batch_counter[:, 0, 1]
                    batch_counter_unique, used_batch_indices = np.unique(batch_counters, return_index=True)

                    for batch_counter, index in zip(batch_counter_unique, used_batch_indices):
                        classIds = classes[index]
                        if self.dynamically_map_classes:
                            classIds = np.array(self._filter_classes.filter_classes(classIds), dtype=np.int64)
                        point = points[index]
                        dist = dists[index]
                        yield point, dist, classIds, [batch_counter], self._start_latent_vector

                    self._batch_counter = np.unique(batch_counters).shape[0]
                    batch_map = np.array(file["batch_counter_map"])
                    self._batch_map = {int(ele[0]): list(ele[1:]) for ele in batch_map}
                elif "points" in file.keys():
                    use_blocker_to_create_blocker_file = True
            if use_blocker_to_create_blocker_file:
                blocker_bin_file_path = Path(__file__).parent.parent.parent.parent / "data_generation" / "tsdf_compression" / "Blocker" / "build" / "Blocker"
                if not blocker_bin_file_path.exists():
                    raise FileNotFoundError("The blocker bin file was not found, build it first! "
                                            "Check data_generation/tsdf_compression/README.md!")

                blocked_tsdf_file_path = convert_tsdf_to_blocked_tsdf_file_path(Path(path), blocker_bin_file_path,
                                                                                only_non_boundary=only_non_boundary)
                if blocked_tsdf_file_path and os.path.exists(blocked_tsdf_file_path):
                    for data in self.get_extract_batches(blocked_tsdf_file_path):
                        yield data
                    os.remove(blocked_tsdf_file_path)

        else:
            use_blocker_to_create_blocker_file = False
            if given_similarity_file_path is not None:
                with h5py.File(path, "r") as file:
                    if "points_dist" in file.keys():
                        point_dist = np.array(file["points_dist"])
                        points = point_dist[:, :, :3].copy()
                        dists = point_dist[:, :, 3].copy()
                        with h5py.File(given_similarity_file_path, "r") as similarity_file:
                            batch_counter_unique = np.array(similarity_file["batch_counter_unique"])
                            used_batch_indices = np.array(similarity_file["used_batch_indices"])
                            check_vectors = np.array(similarity_file["check_vectors"])
                            all_class_ids = np.array(similarity_file["all_class_ids"])
                        self._batch_counter = batch_counter_unique.shape[0]
                        batch_map = np.array(file["batch_counter_map"])
                        self._batch_map = {int(ele[0]): list(ele[1:]) for ele in batch_map}
                        counter = 0
                        print(batch_counter_unique.shape)
                        for batch_counter, index in zip(batch_counter_unique, used_batch_indices):
                            point = points[index]
                            dist = dists[index]
                            class_ids = all_class_ids[counter]
                            if self._return_start_latent_vector:
                                start_latent_vector = self.sim_setter.find_most_similar_for_vector(check_vectors[counter])
                                yield point, dist, class_ids, [batch_counter], start_latent_vector
                            else:
                                yield point, dist, class_ids, [batch_counter]
                            counter += 1

                    elif "points" in file.keys():
                        use_blocker_to_create_blocker_file = True
            else:
                use_blocker_to_create_blocker_file = True
            if use_blocker_to_create_blocker_file:
                blocker_bin_file_path = Path(__file__).parent.parent.parent.parent / "data_generation" / "tsdf_compression" / "Blocker" / "build" / "Blocker"
                if not blocker_bin_file_path.exists():
                    raise FileNotFoundError("The blocker bin file was not found, build it first! "
                                            "Check data_generation/tsdf_compression/README.md!")
                blocked_tsdf_file_path = convert_tsdf_to_blocked_tsdf_file_path(Path(path),
                                                                                blocker_bin_file_path, only_non_boundary=only_non_boundary)
                if blocked_tsdf_file_path and os.path.exists(blocked_tsdf_file_path):
                    similarity_file_path = convert_blocked_to_similarities(blocked_tsdf_file_path,
                                                                           self.sim_setter.database_folder)
                    if os.path.exists(similarity_file_path):
                        for data in self.get_extract_batches(blocked_tsdf_file_path, similarity_file_path):
                            yield data
                        os.remove(similarity_file_path)
                    os.remove(blocked_tsdf_file_path)

    def get_data_point(self):
        self._batch_counter = 0
        if len(self.paths) == 0:
            raise Exception("No data point found!")
        for path in self.paths:
            for data in self.get_extract_batches(path, only_non_boundary=self._only_non_boundary_mode):
                yield data

    def set_paths(self, paths):
        self.paths = paths

    def init_dataset(self):
        output_shape = ((self._point_amount, 3), self._point_amount, self._point_amount, (1,), (self._latent_dim, ))
        output_type = (tf.float32, tf.float32, tf.uint8, tf.int32, tf.float32)
        self.dataset = tf.data.Dataset.from_generator(self.get_data_point, output_shapes=output_shape,
                                                      output_types=output_type)

    def finalize_dataset(self, dataset, repeat=True, shuffle_size=None):
        if dataset is None:
            raise Exception("Call init_dataset before!")
        if repeat:
            dataset = dataset.repeat()
        if shuffle_size is None and self._shuffle_size > 0:
            dataset = dataset.shuffle(self._shuffle_size)
        elif shuffle_size:
            dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.batch(self.batch_size)
        auto_tune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(auto_tune)
        return dataset


class DataSetLoaderTraining(DataSetLoader):

    def __init__(self, settings_manager: SettingsManager):
        super(DataSetLoaderTraining, self).__init__(settings_manager)
        tf_folder_path = settings_manager("DataLoader/tf_folder_path")
        if not os.path.exists(tf_folder_path):
            raise Exception(f"Generate the tf records for this {self.batch_size} and {self._point_amount}: {tf_folder_path}")
        tf_record_files = glob.glob(os.path.join(tf_folder_path, "*.tfrecord"))
        tf_record_files.sort()
        if len(tf_record_files) < 2:
            raise Exception("You need at least two files for training one for validation the rest for training: {len(tf_record_files)}")
        validation_file = tf_record_files[0]
        print("Use as validation file: {}".format(validation_file))
        tf_record_files = tf_record_files[1:]

        should_shuffle = self._shuffle_size > 0
        self.dataset = self.deserialize_dataset_from(tf_record_files, should_shuffle)
        self.validation_dataset = self.deserialize_dataset_from([validation_file], should_shuffle=False)

        self.dataset = self.finalize_dataset(self.dataset)
        self.validation_dataset = self.finalize_dataset(self.validation_dataset, repeat=False, shuffle_size=0)


    def deserialize_dataset_from(self, paths, should_shuffle: bool):
        def read_tfrecord(serialized_example):
            feature_description = {
                'points': tf.io.FixedLenFeature((), tf.string),
                'dists': tf.io.FixedLenFeature((), tf.string),
                'classes': tf.io.FixedLenFeature((), tf.string),
                'batch_counter': tf.io.FixedLenFeature((), tf.int64),
            }
            example = tf.io.parse_single_example(serialized_example, feature_description)

            feature0 = tf.io.parse_tensor(example['points'], out_type=tf.float32)
            feature1 = tf.io.parse_tensor(example['dists'], out_type=tf.float32)
            feature2 = tf.io.parse_tensor(example['classes'], out_type=tf.uint8)
            feature3 = tf.expand_dims(tf.cast(example['batch_counter'], tf.int32), axis=-1)

            return feature0, feature1, feature2, feature3

        dataset_of_file_paths = tf.data.Dataset.from_tensor_slices(paths)
        if should_shuffle:
            dataset_of_file_paths = dataset_of_file_paths.shuffle(len(paths) + 2)
            tfrecord_dataset = dataset_of_file_paths.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
                cycle_length=2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            tfrecord_dataset = tf.data.TFRecordDataset(dataset_of_file_paths, compression_type="GZIP")
        return tfrecord_dataset.map(read_tfrecord)
