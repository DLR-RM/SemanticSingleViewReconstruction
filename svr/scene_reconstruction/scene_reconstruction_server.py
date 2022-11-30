import sys
from pathlib import Path
import socket
import threading
import os
import time
from typing import Optional

import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # block out all tensorflow messages

svr_folder = Path(__file__).parent.parent
sys.path.append(str(svr_folder.absolute()))

from svr.crisscross import Server, ImageMessage, EncodedSceneMessage
from svr.u_net_normal.normal_reconstruction_client import NormalReconstructionClient

from svr.scene_reconstruction.model.tree_model import TreeModel
from svr.scene_reconstruction.utility.util import limit_gpu_usage
from svr.scene_reconstruction.utility.settings_manager import SettingsManager

class SceneReconstructionServer(Server):

    def __init__(self, online: bool = True):
        limit_gpu_usage()
        settings_folder = Path(__file__).parent / "settings"
        settings_path = settings_folder / "settings.yaml"
        app_config_path = settings_folder / "app_config.yaml"
        self.settings_manager = SettingsManager(settings_path, app_config_path, start_logging=False)
        self.settings_manager.set_element("Training/batch_size", 1)
        self.empty_scene = np.zeros((16, 16, 16, 512))

        self.tree_model = TreeModel(settings_manager=self.settings_manager)
        ckpt_file = Path(__file__).parent.parent.parent / "trained_models" / "scene_reconstruction" / "checkpoint.ckpt"
        self.tree_model.load_weights(str(ckpt_file)).expect_partial()
        self.tree_lock = threading.Lock()
        print("Loaded scene reconstruction model")
        self._online = online
        if online:
            super().__init__(server_ip=socket.gethostname(), server_port=1782)
            self.used_target_function = SceneReconstructionServer.on_new_client
            print("Started server")

    def predict_encoded_scene(self, image: np.ndarray, normal_input: Optional[np.ndarray] = None):
        with self.tree_lock:
            start_time = time.time()
            if not isinstance(image, np.ndarray):
                print("Return empty scene, image must be a np.ndarray")
                return self.empty_scene
            if normal_input is None:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # only the color image was provided!
                    client = NormalReconstructionClient()
                    color_img = image
                    normal_img = client.get_normal_img(image)
                    if np.min(normal_img) == 0 and np.max(normal_img) == 0:
                        print("Return empty scene: no normal image generated")
                        return self.empty_scene
                elif len(image.shape) == 3 and image.shape[2] == 6:
                    color_img = image[:, :, :3]
                    normal_img = image[:, :, 3:]
                else:
                    return self.empty_scene
            else:
                color_img = image
                normal_img = normal_input

            if color_img.shape[0] != 512 or color_img.shape[1] != 512:
                color_img = cv2.resize(color_img, (512, 512))
            if np.max(color_img) > 2:
                color_img = color_img.astype(np.float32) / 255.0

            color = np.flip(np.transpose(color_img, axes=[1, 0, 2]), axis=1)
            normal = np.flip(np.transpose(normal_img, axes=[1, 0, 2]), axis=1)
            combined = np.concatenate([color, normal], axis=-1)
            input_for_the_network = combined.reshape((1, combined.shape[0], combined.shape[1], combined.shape[2]))
            prediction, _ = self.tree_model(input_for_the_network, training=False)
            print(f"Tree Scene prediction took: {time.time() - start_time}")
        return prediction.numpy()[0]

    def on_new_client(self, connection: socket.socket, addr):
        """
        For each new client this function gets called it receives the data from the client and sends the answer back.

        This general server can only respond with a mirroring of the content

        :param connection: Open connection to the client
        :param addr: Addr info from the open socket call
        """
        data = self._receive_data(connection)
        if data is not None:
            if isinstance(data, ImageMessage):
                encoded_scene = self.predict_encoded_scene(data.content)
                msg = EncodedSceneMessage(encoded_scene)
                s = time.time()
                self._send_data(connection, msg)
                print(time.time() - s)
            else:
                self._send_data(connection, "unknown command")
        self.close_connection(connection)



if __name__ == "__main__":
    scene_reconstruction_server = SceneReconstructionServer(online=True)
    scene_reconstruction_server.run()




