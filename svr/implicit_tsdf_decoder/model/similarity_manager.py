import glob
import os.path
import subprocess
import h5py
import numpy as np
import math
import time




def convert_file_to_new_one(file_path: str) -> str:
    goal_path = file_path.replace(".hdf5", "_tsdf_blocked.hdf5")
    if not os.path.exists(goal_path):
        blocker_bin_path = "/home/max/workspace/TSDFPointCompression/Blocker/cmake-build-release/Blocker"
        cmd = f"{blocker_bin_path} --path {file_path} --goal_path {goal_path}"
        out = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if out.stderr:
            print(file_path)
            print(out.stderr)
    if os.path.exists(goal_path):
        return goal_path
    else:
        return None


def load_data(file_path: str, filter_cls: "FilterClasses"):
    with h5py.File(file_path, "r") as file:
        if "points_dist" in file.keys():
            point_dist = np.array(file["points_dist"])
            points = point_dist[:, :, :3].copy()
            dists = point_dist[:, :, 3].copy()
            class_batch_counter = np.array(file["class_batch_counter"], dtype=np.int64)
            classes = class_batch_counter[:, :, 0]
            classes = np.array([filter_cls.filter_classes(classIds) for classIds in classes], dtype=np.int64)
            batch_counters = class_batch_counter[:, :, 1]
            return points, dists, classes, batch_counters

def calculate_similarity_vec(points: np.ndarray, dists: np.ndarray, classes: np.ndarray) -> np.ndarray:
    resolution = 4
    res_squared = resolution * resolution
    voxel_size = 2.2 / resolution
    ids = np.floor((points + 1) / (voxel_size)).astype(np.int32)
    ids[:, 1] *= resolution
    ids[:, 2] *= resolution * resolution
    ids = np.sum(ids, axis=1)
    uniques = np.unique(ids)
    final_vec = np.zeros((resolution, resolution, resolution), dtype=np.float64)
    for u in uniques:
        z = math.floor(u / res_squared)
        x, y = u % resolution, math.floor((u - (z * res_squared)) / resolution),
        selection = ids == u
        # dist_mean = np.mean(dists[np.where(selection)])
        # dist_mean = np.mean(dists[selection])
        dist_mean = 0.0
        counter = 1.0
        index = 0
        for selection_val in selection:
            if selection_val:
                inner_fac = 1.0 / counter
                dist_mean = inner_fac * dists[index] + (1.0 - inner_fac) * dist_mean
                counter += 1.0
            index += 1
        # dist_mean = np.mean(np.array(dists[selection]))
        # class_mean = np.mean(np.array(classes[selection]))
        final_vec[x, y, z] = dist_mean
        # final_vec[x, y, z, 1] = class_mean

    final_vec = final_vec.reshape(-1)
    truncation_threshold = 0.8
    final_vec /= truncation_threshold
    final_vec /= len(final_vec)
    class_res = np.zeros(10)
    for class_index in classes:
        class_res[class_index] += 1
    return np.concatenate((final_vec, class_res / classes.shape[0]))

def calculate_similarity_vec_non_boundary(points: np.ndarray, dists: np.ndarray, classes: np.ndarray) -> np.ndarray:
    resolution = 4
    truncation_threshold = 0.8
    res_squared = resolution * resolution
    voxel_size = 2.2 / resolution
    ids = np.floor((points + 1) / (voxel_size)).astype(np.int32)
    ids[:, 1] *= resolution
    ids[:, 2] *= resolution * resolution
    ids = np.sum(ids, axis=1)
    uniques = np.unique(ids)
    final_vec = np.zeros((resolution, resolution, resolution), dtype=np.float64)
    avg_dist = np.mean(dists)
    most_likely_empty_dist = truncation_threshold if avg_dist > 0.0 else -truncation_threshold
    for u in uniques:
        z = math.floor(u / res_squared)
        x, y = u % resolution, math.floor((u - (z * res_squared)) / resolution),
        selection = ids == u
        # dist_mean = np.mean(dists[np.where(selection)])
        # dist_mean = np.mean(dists[selection])
        dist_mean = 0.0
        counter = 1.0
        index = 0
        for selection_val in selection:
            if selection_val:
                inner_fac = 1.0 / counter
                dist_mean = inner_fac * dists[index] + (1.0 - inner_fac) * dist_mean
                counter += 1.0
            index += 1
        # dist_mean = np.mean(np.array(dists[selection]))
        # class_mean = np.mean(np.array(classes[selection]))
        if index == 0:
            # no value found in that selection, so we assume it must be so far away from everything else
            # so we use the most likely empty dist value
            final_vec[x, y, z] = most_likely_empty_dist
        else:
            final_vec[x, y, z] = dist_mean
        # final_vec[x, y, z, 1] = class_mean

    final_vec = final_vec.reshape(-1)
    final_vec /= truncation_threshold
    final_vec /= len(final_vec)
    class_res = np.zeros(10)
    for class_index in classes:
        class_res[class_index] += 1
    return np.concatenate((final_vec, class_res / classes.shape[0]))



def calculate_simalarities_for_points(points: np.ndarray, dists: np.ndarray, classes: np.ndarray):
    sim_times = []
    similarities = []
    for point, dist, class_ids in zip(points, dists, classes):
        start_time = time.time()
        similarity = calculate_similarity_vec(point, dist, class_ids)

        similarities.append(similarity)
        sim_times.append(time.time() - start_time)
    print("Took time per element: {}".format(np.mean(sim_times[2:])))
    return similarities
