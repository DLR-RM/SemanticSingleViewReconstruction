import sys
from pathlib import Path
import socket
import time

import numpy as np
import tensorflow as tf
import cv2
import threading

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.u_net_normal.normal_reconstruction import Model, SettingsManager, limit_gpu_usage
from svr.crisscross import Server, ImageMessage


class NormalReconstructionServer(Server):

    def __init__(self, online: bool = True):
        data_folder = Path(__file__).parent.parent.parent / "trained_models" / "u_net_normal"

        ckpt_file = data_folder / "checkpoint.ckpt"

        settings_file = Path(__file__).parent / "settings" / "settings.yaml"
        app_config_path = settings_file.parent / "app_config.yaml"
        # limit the used gpu usage
        limit_gpu_usage()

        self.settings_manager = SettingsManager(settings_file, app_config_path, start_logging=False)

        self.model = Model(self.settings_manager)
        self.model.build(input_shape=(self.settings_manager("Training/batch_size"), 512, 512, 3))
        self.model.load_weights(ckpt_file)
        self.model_lock = threading.Lock()
        print("Loaded surface normal reconstruction model")
        # should be done after the loading of the model is done
        self._online = online
        if online:
            super().__init__(server_port=1563, server_ip=socket.gethostname())
            self.used_target_function = NormalReconstructionServer.on_new_client
            print("Started server")

    def predict_for_img(self, color_img: np.ndarray):
        with self.model_lock:
            start_time = time.time()
            if color_img.dtype == np.uint8:
                color_img = color_img.astype(np.float32) / 255.0
            if len(color_img.shape) != 3 or color_img.shape[0] != 512 or color_img.shape[1] != 512 or color_img.shape[
                2] != 3:
                if len(color_img.shape) == 3 and color_img.shape[2] == 3 and \
                        (color_img.shape[0] != 512 or color_img.shape[1] != 512):
                    color_img = cv2.resize(color_img, (512, 512))
                else:
                    return np.zeros((512, 512, 3))

            tf_image = tf.expand_dims(tf.reverse(tf.transpose(color_img, perm=(1, 0, 2)), axis=[1]), axis=0)
            prediction = self.model(tf_image, training=True).numpy()[0]
            prediction = tf.transpose(tf.reverse(prediction, axis=[1]), perm=(1, 0, 2))

        print(f"Surface normal prediction took: {time.time() - start_time}s")
        return prediction

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
                normal_img = self.predict_for_img(color_img=data.content)
                msg = ImageMessage(normal_img)
                self._send_data(connection, msg)
            else:
                self._send_data(connection, "unknown command")
        self.close_connection(connection)


if __name__ == "__main__":
    normal_reconstruction_server = NormalReconstructionServer()

    normal_reconstruction_server.run()
