from pathlib import Path
from typing import List
import os

import numpy as np
import h5py
import pickle
from scipy.spatial import cKDTree
from numba import njit

from data_generation.tsdf_compression.src.convert_single_file import predict_for_file
from data_generation.tsdf_compression.src.convert_single_file_non_boundary import \
    predict_for_file as predict_for_file_non_boundary

from svr.implicit_tsdf_decoder.model.similarity_manager import load_data, calculate_similarity_vec, calculate_similarity_vec_non_boundary
from svr.implicit_tsdf_decoder.model.filter_classes import FilterClasses
from svr.implicit_tsdf_decoder.model.convert_tsdf_to_blocked_tsdf_file import convert_tsdf_to_blocked_tsdf_file_path


def predict_boundary_voxels(path: Path, output_folder: Path, set_start_latent_vector=False, database_folder: Path = None) -> Path:
    folder_name = path.parent.name
    output_file = output_folder / folder_name / f'{path.with_suffix("").name}_db_boundary.hdf5'
    if not output_file.exists():
        predict_for_file(path, output_file, set_start_latent_vector=set_start_latent_vector,
                         database_folder=database_folder)
    return output_file


def predict_non_boundary_voxels(path: Path, output_folder: Path, set_start_latent_vector=False, database_folder: Path = None) -> Path:
    folder_name = path.parent.name
    output_file = output_folder / folder_name / f'{path.with_suffix("").name}_db_non_boundary.hdf5'
    if not output_file.exists():
        predict_for_file_non_boundary(path, output_file, set_start_latent_vector=set_start_latent_vector,
                                      database_folder=database_folder)
    return output_file


def combine_to_database(tsdf_file_paths: List[Path], final_database_path: Path,
                        blocker_bin_file_path: Path, non_boundary: bool = False):
    filter_classes = FilterClasses()
    if non_boundary:
        calculate_similarity_vec_used = njit(calculate_similarity_vec_non_boundary)
    else:
        calculate_similarity_vec_used = njit(calculate_similarity_vec)
    giant_latent_vector = []
    giant_similarity_vector = []
    tsdf_file_paths = [path for path in tsdf_file_paths if path.exists()]
    for tsdf_file_path in tsdf_file_paths:
        with h5py.File(tsdf_file_path, "r") as file:
            if "tsdf_file_path" in file.keys() and "latent_vec" in file.keys():
                file_path = Path(str(np.array(file["tsdf_file_path"])).replace("b'", "")[:-1])
                if not file_path.exists():
                    continue
                created_block_file = convert_tsdf_to_blocked_tsdf_file_path(file_path, blocker_bin_file_path, non_boundary)
                points, dists, classes, batch_counters = load_data(created_block_file, filter_classes)
                os.remove(created_block_file)
                similarities = []
                unique, indices = np.unique(batch_counters[:, 0], return_index=True)
                for batch_counter in indices:
                    similarities.append(calculate_similarity_vec_used(points[batch_counter], dists[batch_counter], classes[batch_counter]))
                latent_vectors = np.array(file["latent_vec"])
                giant_latent_vector.extend(list(latent_vectors))
                if latent_vectors.shape[0] < len(similarities):
                    similarities = similarities[:latent_vectors.shape[0]]
                giant_similarity_vector.extend(similarities)

    giant_latent_vector = np.array(giant_latent_vector)
    giant_similarity_vector = np.array(giant_similarity_vector)
    print("Build tree:")
    tree = cKDTree(giant_similarity_vector)
    print("Done building tree")

    final_database_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_database_path, "wb") as file:
        pickle.dump((tree, giant_similarity_vector, giant_latent_vector), file)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Create the tsdf compression database")
    parser.add_argument("tsdf_point_cloud_folder", help="Path to the tsdf point clouder folder")
    parser.add_argument("database_output_folder", help="Path to the database output folder")
    parser.add_argument("--database_size", help="Used size for the database greater values, take longer to predict the "
                                                "database, but speed up the prediction in the end. A value around 1000 "
                                                "seems to work well.", default=1000, type=int)
    args = parser.parse_args()

    tsdf_point_cloud_folder = Path(args.tsdf_point_cloud_folder)
    if not tsdf_point_cloud_folder.exists():
        raise FileNotFoundError("The tsdf point cloud folder does not exist!")
    blocker_bin_file = Path(__file__).parent / "Blocker" / "build" / "Blocker"
    if not blocker_bin_file.exists():
        raise FileNotFoundError("The blocker bin file was not found, build it first! "
                                "Check data_generation/tsdf_compression/README.md!")

    database_output_folder = Path(args.database_output_folder)
    database_output_folder.mkdir(parents=True, exist_ok=True)

    hdf5_files = list(tsdf_point_cloud_folder.glob("**/*.hdf5"))
    hdf5_files = [hdf5_file for hdf5_file in hdf5_files if "_loss_avg.hdf5" not in hdf5_file.name]

    print(f"Found {len(hdf5_files)} hdf5 files, use only {args.database_size}!")

    boundary_files, non_boundary_files = [], []
    for hdf5_file in hdf5_files[:args.database_size]:
        boundary_files.append(predict_boundary_voxels(hdf5_file, database_output_folder))
        non_boundary_files.append(predict_non_boundary_voxels(hdf5_file, database_output_folder))

    combine_to_database(boundary_files, database_output_folder / "boundary.pickle", blocker_bin_file,
                        non_boundary=False)
    combine_to_database(boundary_files, database_output_folder / "non_boundary.pickle", blocker_bin_file, non_boundary=True)



