import os
from pathlib import Path
import subprocess

from git import Repo


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Render all color and normal images with randomized texture images for the 3D "
                                     "Front dataset.")
    parser.add_argument('front_json_folder', help="3D FRONT path to the folder containing the json files")
    parser.add_argument('future_model_path', help="3D Future model path")
    parser.add_argument('front_3D_texture_path', help="Path to the 3D Front texture path")
    parser.add_argument('output_image_folder', help="Output folder for the images")
    parser.add_argument('output_front_3D_obj_mesh_folder', help="Front 3D folder in which the meshes of the objects are saved.")
    parser.add_argument('cc_textures', help="Path to the cctextures folder.")
    args = parser.parse_args()

    render_script = Path(__file__).parent / "render_script.py"
    main_folder = Path(__file__).parent.parent.parent.absolute()

    blenderproc_download_folder = Path(__file__).parent / "open_source_blenderproc"
    if not blenderproc_download_folder.exists():
        blenderproc_download_folder.mkdir(parents=True, exist_ok=True)
        Repo.clone_from("https://github.com/DLR-RM/BlenderProc.git", blenderproc_download_folder.absolute())

    cli_script = blenderproc_download_folder / "cli.py"
    if not cli_script.exists():
        raise FileNotFoundError(f"The cli script could not be found try manually deleting the "
                                f"{blenderproc_download_folder} and try again.")

    front_3d_json_folder = Path(args.front_json_folder)
    if not front_3d_json_folder.exists():
        raise FileNotFoundError("The 3D Front json folder does not exist!")

    future_model_path = Path(args.future_model_path)
    if not future_model_path.exists():
        raise FileNotFoundError("The future model path does not exist!")

    front_3d_texture_folder = Path(args.front_3D_texture_path)
    if not front_3d_texture_folder.exists():
        raise FileNotFoundError("The front 3d texture path does not exist!")

    cc_textures_folder = Path(args.cc_textures)
    if not cc_textures_folder.exists():
        raise FileNotFoundError("The cc texture folder path does not exist!")

    json_files = list(front_3d_json_folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No 3D Front json files could be found!")

    output_img_folder = Path(args.output_image_folder).absolute()
    output_img_folder.mkdir(parents=True, exist_ok=True)

    output_obj_folder = Path(args.output_front_3D_obj_mesh_folder).absolute()
    output_obj_folder.mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        current_name = json_file.with_suffix("").name
        current_output_img_folder = output_img_folder / current_name
        if current_output_img_folder.exists():
            # skip folders which have been generated before
            continue
        current_output_img_folder.mkdir(parents=True, exist_ok=True)

        render_cmd = f"python {cli_script} run {render_script} {json_file} {future_model_path} {front_3d_texture_folder} " \
                     f"{current_output_img_folder} {output_obj_folder} {cc_textures_folder}"
        subprocess.run(render_cmd, shell=True, cwd=str(blenderproc_download_folder.absolute()))





