import socket
import sys
from pathlib import Path

import numpy as np

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.crisscross import Client, EncodedSceneMessage, SceneMessage


class ImplicitTsdfClient(Client):

    def __init__(self):
        super().__init__(server_ip=socket.gethostname(), server_port=1863)

    def get_decompressed_scene(self, compressed_scene: np.ndarray) -> np.ndarray:
        msg = EncodedSceneMessage(scene=compressed_scene)
        res = self.get_for_message(msg)
        if res is not None and isinstance(res, SceneMessage):
            return res.content
        else:
            return SceneMessage(np.zeros(256, 256, 256), np.zeros(256, 256, 256), 0.1)



if __name__ == "__main__":
    import argparse
    import h5py

    parser = argparse.ArgumentParser("Convert a single compressed file to a decompressed file")
    parser.add_argument("input_file", help="Path to the input file!")
    parser.add_argument("output_file", help="Path to the output file!")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as file:
        compressed_scene = np.array(file["combined_latent_block"])

    client = ImplicitTsdfClient()
    result = client.get_decompressed_scene(compressed_scene)

    with h5py.File(args.output_file, "w") as file:
        file.create_dataset("voxelblock", data=result["voxel_block"], compression="gzip")
        file.create_dataset("classes", data=result["class_block"], compression="gzip")

