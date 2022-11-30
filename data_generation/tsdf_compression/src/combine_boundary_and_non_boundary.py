from pathlib import Path
from typing import Tuple
import os

import h5py
import numpy as np


def get_filled_and_empty_latent_vec() -> Tuple[np.ndarray, np.ndarray]:
    hdf5_file_path = Path(__file__).parent.parent.parent.parent / "trained_models" / "implicit_tsdf_decoder" \
                     / "empty_and_filled_latent_value.hdf5"
    with h5py.File(hdf5_file_path, "r") as file:
        filled_latent_vec = np.array(file["filled_latent_vec"])
        empty_latent_vec = np.array(file["empty_latent_vec"])
    return filled_latent_vec, empty_latent_vec


def combine_to_one_block(boundary_path: Path, non_boundary_latent_value_path: Path,
                         org_tsdf_file: Path, output_path: Path,):
    if not boundary_path.exists():
        raise Exception("The boundary file does not exist")
    if not non_boundary_latent_value_path.exists():
        raise Exception("The non boundary file does not exist!")
    if not org_tsdf_file.exists():
        raise Exception(f"The org tsdf file does not exist: {org_tsdf_file}!")

    with h5py.File(boundary_path, "r") as file:
        resolution = int(file["resolution"][0])
        latent_locs = np.array(file["latent_locations"])
        latent_vecs = np.array(file["latent_vec"])

    filled_latent_vec, empty_latent_vec = get_filled_and_empty_latent_vec()

    used_blocks = np.zeros([resolution, resolution, resolution], dtype=bool)
    combined_block = np.zeros([resolution, resolution, resolution, latent_vecs.shape[1]])
    for pos, latent_vec in zip(latent_locs, latent_vecs):
        combined_block[pos[0], pos[1], pos[2]] = latent_vec
        used_blocks[pos[0], pos[1], pos[2]] = True

    with h5py.File(non_boundary_latent_value_path, "r") as file:
        if resolution != int(file["resolution"][0]):
            raise Exception("The resolution is not the same")
        non_boundary_latent_locs = np.array(file["latent_locations"])
        non_boundary_latent_vecs = np.array(file["latent_vec"])

    for pos, latent_vec in zip(non_boundary_latent_locs, non_boundary_latent_vecs):
        combined_block[pos[0], pos[1], pos[2]] = latent_vec
        used_blocks[pos[0], pos[1], pos[2]] = True

    with h5py.File(org_tsdf_file, "r") as file:
        voxel_space = np.array(file["voxel_space"], dtype=np.double) / np.iinfo(np.uint16).max * 2 - 1
    m_size = voxel_space.shape[0] // resolution
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                if not used_blocks[x, y, z]:
                    mean_value = np.mean(voxel_space[m_size * x: m_size * (x + 1), m_size * y: m_size * (y + 1),
                                         m_size * z: m_size * (z + 1)])
                    if mean_value < 0:
                        # use the filled latent vec
                        combined_block[x, y, z] = filled_latent_vec
                    else:
                        # use the empty latent vec
                        combined_block[x, y, z] = empty_latent_vec

    if not output_path.parent.exists():
        os.makedirs(output_path.parent, exist_ok=True)

    with h5py.File(output_path, "w") as file:
        file.create_dataset(f"combined_latent_block", data=combined_block, compression="gzip")
        file.create_dataset("resolution", data=np.array([resolution]))
        org_tsdf_file_path = str(org_tsdf_file)
        file.create_dataset("tsdf_file_path", data=np.string_(org_tsdf_file_path),
                            dtype=np.string_(org_tsdf_file_path).dtype)
        latent_tsdf_file = str(boundary_path)
        file.create_dataset("latent_value_tsdf_file", data=np.string_(latent_tsdf_file),
                            dtype=np.string_(latent_tsdf_file).dtype)
        non_boundary_latent_tsdf_file = str(non_boundary_latent_value_path)
        file.create_dataset("non_boundary_latent_value_tsdf_file", data=np.string_(non_boundary_latent_tsdf_file),
                            dtype=np.string_(non_boundary_latent_tsdf_file).dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Combine single file to one block")
    parser.add_argument("--boundary_path", help="Path to the boundary hdf5 file", required=True)
    parser.add_argument("--non_boundary_path", help="Path to the non boundary hdf5 file", required=True)
    parser.add_argument("--tsdf_file", help="Path to the tsdf file", required=True)
    parser.add_argument("--output_path", help="Path to the output hdf5 file", required=True)
    args = parser.parse_args()

    combine_to_one_block(Path(args.boundary_path), Path(args.non_boundary_path), Path(args.tsdf_file),
                         Path(args.output_path))
