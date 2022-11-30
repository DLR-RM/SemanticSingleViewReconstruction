import argparse
from pathlib import Path
import multiprocessing
import subprocess
import threading

def create_loss_map():
    global hdf5_files

    while len(hdf5_files) > 0:
        current_path = hdf5_files.pop()
        output_file = current_path.parent / f"{current_path.with_suffix('').name}_loss_avg.hdf5"
        if output_file.exists():
            continue
        cmd = f"{losscalc_file} -p {current_path} -r 256"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create the loss maps for the tsdf point clouds, they will be generated next "
                                     "to the tsdf files.")
    parser.add_argument("tsdf_point_cloud_folder", help="Path to the tsdf point cloud folder")
    args = parser.parse_args()

    tsdf_point_cloud_folder = Path(args.tsdf_point_cloud_folder)
    if not tsdf_point_cloud_folder.exists():
        raise FileNotFoundError("The input folder does not exist!")

    losscalc_file = Path(__file__).parent / "build" / "LossCalculator"
    if not losscalc_file.exists():
        raise FileNotFoundError("The LossCalculator has to be build before use, please read the README.md")

    hdf5_files = list(tsdf_point_cloud_folder.glob("**/*.hdf5"))
    hdf5_files = [hdf5_file for hdf5_file in hdf5_files if "_loss_avg.hdf5" not in hdf5_file.name]
    print(f"Found {len(hdf5_files)} hdf5 files")

    amount_of_threads = multiprocessing.cpu_count()

    processes = []
    for i in range(amount_of_threads):
        processes.append(threading.Thread(target=create_loss_map))
        processes[-1].daemon = True
        processes[-1].start()

    for p in processes:
        p.join()



