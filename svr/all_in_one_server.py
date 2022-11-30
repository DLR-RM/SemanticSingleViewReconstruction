import sys
import time
from pathlib import Path
import socket

import cv2
import numpy as np

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.u_net_normal.normal_reconstruction_server import NormalReconstructionServer
from svr.scene_reconstruction.scene_reconstruction_server import SceneReconstructionServer
from svr.implicit_tsdf_decoder.implicit_tsdf_decoder_server import ImplicitTsdfDecoderServer
from svr.crisscross import Server, ImageMessage, SceneMessage


class AllInOneServer(Server):

    def __init__(self):
        self.normal_server = NormalReconstructionServer(online=False)
        self.tree_server = SceneReconstructionServer(online=False)
        self.decoder_server = ImplicitTsdfDecoderServer(online=False)
        print("Loaded all models")
        super().__init__(server_port=3163, server_ip=socket.gethostname())
        self.used_target_function = AllInOneServer.on_new_client
        print("Started server")

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
                color_img = data.content
                if color_img.shape[0] != 512 or color_img.shape[1] != 512:
                    color_img = cv2.resize(color_img, (512, 512))
                start_time = time.time()
                normal_img = self.normal_server.predict_for_img(color_img=color_img)
                if np.min(normal_img) == 0 and np.max(normal_img) == 0:
                    print("No normal image reconstructed!")
                    return SceneMessage(np.zeros(256, 256, 256), np.zeros(256, 256, 256),
                                        self.decoder_server.truncation_threshold)
                encoded_scene = self.tree_server.predict_encoded_scene(color_img, normal_img)
                if np.min(encoded_scene) == 0 and np.max(encoded_scene) == 0:
                    print("No encoded scene reconstructed!")
                voxel_output, class_output = self.decoder_server.create_tsdf_block(encoded_scene)
                print(f"Full prediction took: {time.time() - start_time}")
                msg = SceneMessage(voxel_output, class_output, self.decoder_server.truncation_threshold)
                self._send_data(connection, msg)
            else:
                self._send_data(connection, "unknown command")
        self.close_connection(connection)


if __name__ == "__main__":
    all_in_one_server = AllInOneServer()
    # start server
    all_in_one_server.run()
