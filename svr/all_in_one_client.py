import socket
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
import numpy as np

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.crisscross import Client, ImageMessage, SceneMessage
from svr.implicit_tsdf_decoder import convert_final_block_to_mesh, generate_image_mtl, generate_mtl_file


class AllInOneClient(Client):

    def __init__(self, server_ip: Optional[str]):
        if server_ip is None:
            server_ip = socket.gethostname()
        super().__init__(server_ip=server_ip, server_port=3163)

    def get_scene(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        msg = ImageMessage(img)
        t = time.time()
        res: SceneMessage = self.get_for_message(msg)
        print(f"Server call took: {time.time() - t}s")
        if res is not None and isinstance(res, SceneMessage):
            voxel = res.content["voxel_block"]
            if voxel.dtype == np.uint16:
                voxel = voxel.astype(np.float32) / np.iinfo(np.uint16).max * 0.1 * 2 - 0.1
            return voxel, res.content["class_block"]
        else:
            return np.zeros((256, 256, 256)), np.zeros((256, 256, 256))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Connect to the all in one server and predict for an image a 3D scene")
    parser.add_argument("input_file_path", help="Path to input file, make sure that the opening angle of the image "
                                                "is correct! Check the svr/README.md for more information!")
    parser.add_argument("output_file_path", help="Path to desired output .obj file")
    parser.add_argument("--add_image", help="If this is true the color image is projected on the scene", default=False,
                        action="store_true")
    parser.add_argument("--server_ip", help="Name of the node, where the server is running on", default=None)
    args = parser.parse_args()

    all_in_one_client = AllInOneClient(args.server_ip)

    input_image_path = Path(args.input_file_path)
    if not input_image_path.exists():
        raise FileNotFoundError("The input file does not exist!")

    output_file_path = Path(args.output_file_path)
    if output_file_path.suffix != ".obj":
        raise RuntimeError("The file type of the output file must be .obj!")

    color_img = Image.open(str(input_image_path))
    test_img = np.asarray(color_img)
    output_block, class_block = all_in_one_client.get_scene(test_img)

    new_text, _ = convert_final_block_to_mesh(output_block, class_block, unproject=True, add_uv_texture=args.add_image)
    text = "mtllib text.mtl\n" + new_text
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w") as file:
        file.write(text)
    if args.add_image:
        color_img.save(str(output_file_path.with_suffix(".png")))
        generate_image_mtl(output_file_path)
    else:
        generate_mtl_file(output_file_path)
