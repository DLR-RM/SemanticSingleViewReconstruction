import argparse

from numba import njit
import numpy as np
import h5py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from svr.implicit_tsdf_decoder.model.filter_classes import FilterClasses
from svr.implicit_tsdf_decoder.model.similarity_manager import calculate_similarity_vec
from svr.implicit_tsdf_decoder.model.similarity_setter import SimilaritySetter


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert blocked file to similarity vector file")
    parser.add_argument("--path", help="Path to the blocked .hdf5 file")
    parser.add_argument("--goal_path", help="Path where the similarities should be saved to.")
    parser.add_argument("database_folder", help="Path to the database folder")
    args = parser.parse_args()

    path = args.path

    dynamically_map_classes = True
    filter_classes = FilterClasses()
    sim_setter = SimilaritySetter(args.database_folder)
    calculate_similarity_vec = njit(calculate_similarity_vec)
    with h5py.File(path, "r") as file:
        if "points_dist" in file.keys():
            point_dist = np.array(file["points_dist"])
            points = point_dist[:, :, :3].copy()
            dists = point_dist[:, :, 3].copy()
            class_batch_counter = np.array(file["class_batch_counter"])
            classes = class_batch_counter[:, :, 0]
            batch_counters = class_batch_counter[:, 0, 1]
            batch_counter_unique, used_batch_indices = np.unique(batch_counters, return_index=True)
            check_vectors = []
            all_class_ids = []
            for batch_counter, index in zip(batch_counter_unique, used_batch_indices):
                 class_ids = classes[index]
                 class_ids = np.array(filter_classes.filter_classes(class_ids), dtype=np.int32)
                 all_class_ids.append(class_ids)
                 point = points[index]
                 dist = dists[index]
                 check_vectors.append(calculate_similarity_vec(point, dist, class_ids))

            all_class_ids = np.array(all_class_ids)
            check_vectors = np.array(check_vectors)
    with h5py.File(args.goal_path, "w") as file:
        file.create_dataset("check_vectors", data=check_vectors)
        file.create_dataset("all_class_ids", data=all_class_ids)
        file.create_dataset("batch_counter_unique", data=batch_counter_unique)
        file.create_dataset("used_batch_indices", data=used_batch_indices)

