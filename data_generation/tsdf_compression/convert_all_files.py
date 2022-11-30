import shutil
from pathlib import Path
import argparse

from data_generation.tsdf_compression.create_database import predict_boundary_voxels, predict_non_boundary_voxels
from data_generation.tsdf_compression.src.combine_boundary_and_non_boundary import combine_to_one_block

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Convert all tsdf point clouds into compressed latent encodings using the database")
    parser.add_argument("tsdf_point_cloud_folder", help="Path to the tsdf point cloud folder")
    parser.add_argument("database_folder", help="Path where the two pickle files are stored")
    parser.add_argument("compressed_output_folder", help="Path to the output folder in which the compressed files "
                                                         "will be stored.")
    args = parser.parse_args()

    tsdf_point_cloud_folder = Path(args.tsdf_point_cloud_folder)
    if not tsdf_point_cloud_folder.exists():
        raise FileNotFoundError("The tsdf point cloud folder does not exist!")
    database_folder = Path(args.database_folder)
    if not database_folder.exists():
        raise FileNotFoundError("The database folder does not exist!")

    output_folder = Path(args.compressed_output_folder)

    hdf5_files = list(tsdf_point_cloud_folder.glob("**/*.hdf5"))
    hdf5_files = [hdf5_file for hdf5_file in hdf5_files if "_loss_avg.hdf5" not in hdf5_file.name]

    print(f"Found {len(hdf5_files)} hdf5 files!")

    temp_output_folder = Path("/tmp/current_output_folder")

    # walk over all hdf5 files and compress them one after the other
    for hdf5_file in hdf5_files:
        output_file = output_folder / hdf5_file.parent.name / hdf5_file.name
        if output_file.exists():
            continue
        temp_output_folder.mkdir(parents=True, exist_ok=True)
        boundary_file = predict_boundary_voxels(hdf5_file, temp_output_folder, set_start_latent_vector=True, database_folder=database_folder)
        non_boundary_file = predict_non_boundary_voxels(hdf5_file, temp_output_folder, set_start_latent_vector=True, database_folder=database_folder)
        combine_to_one_block(boundary_file, non_boundary_file, hdf5_file, output_file)
        shutil.rmtree(temp_output_folder)


