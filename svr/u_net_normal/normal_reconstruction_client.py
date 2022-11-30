import socket
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.crisscross import Client, ImageMessage


class NormalReconstructionClient(Client):

    def __init__(self, server_ip: Optional[str] = None):
        if server_ip is None:
            server_ip = socket.gethostname()
        super().__init__(server_ip=server_ip, server_port=1563)

    def get_normal_img(self, img: np.ndarray) -> np.ndarray:
        msg = ImageMessage(img)
        res = self.get_for_message(msg)
        if res is not None and isinstance(res, ImageMessage):
            return res.content
        else:
            return np.zeros((512, 512, 3))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    client = NormalReconstructionClient()

    demo_folder = Path(__file__).parent.parent.parent / "demo"
    test_img_paths = list(demo_folder.glob("*.jpg"))
    counter = 1
    for test_img_path in test_img_paths:
        # read in color image
        test_img = np.asarray(Image.open(str(test_img_path)))
        # predict surface normals in range -1 to 1
        normal_img = client.get_normal_img(test_img)

        # check if the server is running
        if np.mean(np.abs(normal_img - np.zeros(normal_img.shape))) < 1e-5:
            raise RuntimeError("The server is not started yet!")

        # plot color image
        plt.subplot(len(test_img_paths), 2, counter)
        counter += 1
        plt.title(f"Color of {test_img_path.with_suffix('').name}")
        plt.imshow(test_img)

        # plot normal image
        plt.subplot(len(test_img_paths), 2, counter)
        counter += 1
        plt.title(f"Surface Normal of {test_img_path.with_suffix('').name}")
        # map image to range of 0 to 1
        plt.imshow(normal_img * 0.5 + 0.5)
        print(f"Done with: {test_img_path.with_suffix('').name}")
    plt.show()
