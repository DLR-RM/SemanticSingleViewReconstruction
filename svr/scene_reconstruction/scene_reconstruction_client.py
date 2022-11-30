import socket
import sys
from pathlib import Path
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # block out all tensorflow messages


svr_folder = Path(__file__).parent.parent
sys.path.append(str(svr_folder.absolute()))

from svr.crisscross import Client, ImageMessage, EncodedSceneMessage


class SceneReconstructionClient(Client):

    def __init__(self):
        super().__init__(server_ip=socket.gethostname(), server_port=1782)

    def get_encoded_scene(self, img: np.ndarray) -> np.ndarray:
        msg = ImageMessage(img)
        res = self.get_for_message(msg)
        if res is not None and isinstance(res, EncodedSceneMessage):
            return res.content
        else:
            return np.zeros((16, 16, 16, 512))


if __name__ == "__main__":
    import argparse
    import h5py

    parser = argparse.ArgumentParser("Convert a hdf5 container with color and normal image into a compressed "
                                     "latent representation")
    parser.add_argument("input_file", help="Path to the hdf5 container containing a color and normal image. "
                                           "You can generate these with the data_generation/README.md")
    parser.add_argument("output_file", help="An encoded scene, to decompress this you can use the "
                                            "svr/implicit_tsdf_decoder.")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as file:
        color = np.array(file["colors"])
        normals = np.array(file["normals"])

    client = SceneReconstructionClient()
    img = np.concatenate((color, normals), axis=-1)

    res = client.get_encoded_scene(img)

    with h5py.File(args.output_file, "w") as file:
        file.create_dataset("combined_latent_block", data=res, compression="gzip")

