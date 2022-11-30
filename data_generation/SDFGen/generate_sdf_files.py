import threading
from pathlib import Path
import argparse
import subprocess
import time
import os
import shutil
import multiprocessing


def run_in_parallel():
    global single_camera_pose_files
    while len(single_camera_pose_files) > 0:
        single_camera_pose_file = single_camera_pose_files.pop()
        name = single_camera_pose_file.name
        folder_name = name[:name.find("_")]
        current_goal_folder = goal_folder / folder_name
        current_data_folder = data_folder / folder_name
        run_for_file(sdf_gen_build_file, current_data_folder / "position.txt", current_data_folder / "wall_objects.obj",
                     folder_name, single_camera_pose_file, current_goal_folder)

def run_for_file(sdf_gen_bin_file: Path, not_fixed_position_file: Path, wall_objects_file: Path, folder_name: str,
                 camera_position: Path, tsdf_output_folder: Path):
    tsdf_output_folder.mkdir(parents=True, exist_ok=True)
    position_file = fix_position_file(not_fixed_position_file, tsdf_output_folder, folder_name, camera_position)
    nr = camera_position.name
    nr = nr[nr.rfind("_")+1:]
    current_output_tsdf_folder = tsdf_output_folder / nr
    current_output_tsdf_folder.mkdir(exist_ok=True, parents=True)
    cmd = "{} -p {} -w {} -c {} -f {} --threads 1".format(sdf_gen_bin_file, position_file, wall_objects_file,
                                                           camera_position, current_output_tsdf_folder)
    print(cmd)
    subprocess.call(cmd, shell=True)
    time.sleep(0.5)
    for file_path in current_output_tsdf_folder.glob("*.hdf5"):
        shutil.move(file_path, current_output_tsdf_folder.parent / f"{nr}.hdf5")
    shutil.rmtree(current_output_tsdf_folder)
    os.remove(position_file)


def fix_position_file(position_file: Path, goal_folder: Path, house_id: str, camera_file: Path) -> Path:
    with position_file.open("r") as file:
        position_text = file.read().split("\n")
    position_text = [line.replace(".blend", ".obj") for line in position_text if line]
    nr = camera_file.name
    camera_nr = nr[nr.rfind("_")+1:]
    current_position_file = goal_folder / "{}_{}_position_file.txt".format(house_id, camera_nr)
    with open(current_position_file, "w") as file:
        file.write("\n".join(position_text))
    return current_position_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert the camera poses files into tsdf point clouds.")
    parser.add_argument('front_3D_obj_mesh_folder', help="Front 3D folder in which the meshes of the objects are saved.")
    parser.add_argument('tsdf_point_cloud_folder', help="TSDF point cloud output folder")
    args = parser.parse_args()

    sdf_gen_build_file = Path(__file__).parent / "build" / "sdfgen"
    if not sdf_gen_build_file.exists():
        raise FileNotFoundError("The sdfgen file could not be found, make sure to build the project first!")

    data_folder = Path(args.front_3D_obj_mesh_folder)
    camera_pose_files = list(data_folder.glob("**/camera_positions"))
    if not camera_pose_files:
        raise FileNotFoundError("No camera pose files found!")
    print(f"Found {len(camera_pose_files)} camera pose files")

    goal_folder = Path(args.tsdf_point_cloud_folder)
    temp_folder = Path("/tmp/semantic_svr/")
    temp_folder.mkdir(parents=True, exist_ok=True)
    for camera_pose_file in camera_pose_files:
        with camera_pose_file.open("r") as file:
            for index, line in enumerate(file.read().split("\n")):
                if line:
                    output_file = goal_folder / camera_pose_file.parent.name / f"{index}.hdf5"
                    if output_file.exists():
                        continue
                    new_file_path = temp_folder / f"{camera_pose_file.parent.name}_camera_position_{index}"
                    with new_file_path.open("w") as file2:
                        file2.write(line)

    single_camera_pose_files = list(Path(temp_folder).glob("*_camera_position_*"))
    print(f"Found {len(single_camera_pose_files)} camera poses!")

    amount_of_threads = multiprocessing.cpu_count()

    threads = []
    for i in range(amount_of_threads):
        thread = threading.Thread(target=run_in_parallel)
        thread.daemon = True
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    shutil.rmtree(temp_folder)




