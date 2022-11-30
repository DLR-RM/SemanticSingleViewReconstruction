
import os
import subprocess
from pathlib import Path


def convert_tsdf_to_blocked_tsdf_file_path(tsdf_file_path: Path, block_bin_file_path: Path,
                                           only_non_boundary: bool = False) -> str:
    if tsdf_file_path.exists():
        room_id = tsdf_file_path.parent.name
        file_name = tsdf_file_path.name.replace(".hdf5", "_blocked.hdf5")
        goal_path = os.path.join("/dev/shm", room_id + "_" + ("boundary_" if only_non_boundary else "") + file_name)
        cmd = f"{block_bin_file_path} -p {tsdf_file_path} -g {goal_path}"
        if only_non_boundary:
            cmd += " --block_selection_mode NON-BOUNDARY"
        subprocess.check_output(cmd, shell=True)
        return goal_path


def convert_blocked_to_similarities(blocked_tsdf_file_path: str, database_folder: Path) -> str:
    if os.path.exists(blocked_tsdf_file_path):
        goal_path = blocked_tsdf_file_path.replace("_blocked.hdf5", "_similarities.hdf5")
        file_path = os.path.join(os.path.dirname(__file__), "create_similarities_for_each_file.py")
        cmd = f"python {file_path} --path {blocked_tsdf_file_path} --goal_path {goal_path} {database_folder.absolute()}"
        subprocess.check_output(cmd, shell=True)
        return goal_path

