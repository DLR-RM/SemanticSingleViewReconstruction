from pathlib import Path
import sys
import time
from typing import Dict, Optional
import socket

import tensorflow as tf
import numpy as np
import h5py
import numba

normal_rec_folder = Path(__file__).parent.parent.parent
sys.path.append(str(normal_rec_folder.absolute()))

from svr.implicit_tsdf_decoder.model.trainer import Trainer
from svr.implicit_tsdf_decoder.utility.settings_manager import SettingsManager, limit_gpu_usage
from svr.crisscross import Server, EncodedSceneMessage, SceneMessage


class ImplicitTsdfDecoderServer(Server):

    def __init__(self, online: bool = True):
        trained_folder = Path(__file__).parent.parent.parent / "trained_models"
        settings_path = Path(__file__).parent / "settings" / "settings.yaml"
        app_config_path = settings_path.parent / "app_config.yaml"
        self.weights_path = trained_folder / "implicit_tsdf_decoder" / "cp-0000.ckpt"

        limit_gpu_usage()
        settings_manager = SettingsManager(settings_path, app_config_path, start_logging=False)
        settings_manager.app_data["DataLoader"]["shuffle_size"] = 0
        amount_of_blocks = settings_manager("DataLoader/resolution")

        org_truncation_threshold = settings_manager("DataLoader/org_trunc_threshold")  # fixed in the DataSetLoader
        resolution = settings_manager("DataLoader/resolution")
        self.truncation_threshold = org_truncation_threshold / (2.0 / resolution)
        # add the correct trunc threshold to the settings
        settings_manager.data["Generator"]["trunc_threshold"] = self.truncation_threshold
        self.number_of_classes = settings_manager("Generator/number_of_classes")

        empty_and_filled_file = trained_folder / "implicit_tsdf_decoder" / "empty_and_filled_latent_value.hdf5"
        with h5py.File(empty_and_filled_file, "r") as file:
            self.filled_latent_vec = np.array(file["filled_latent_vec"])
            self.filled_latent_vec = np.tile(self.filled_latent_vec, (amount_of_blocks, amount_of_blocks, amount_of_blocks, 1))
            self.empty_latent_vec = np.array(file["empty_latent_vec"])
            self.empty_latent_vec = np.tile(self.empty_latent_vec, (amount_of_blocks, amount_of_blocks, amount_of_blocks, 1))
        self.output_resolution = 256
        self.info = {}
        self.trainer: Dict[int, Trainer] = {}
        self.small_res: int = 8
        self.big_res = self.output_resolution // amount_of_blocks
        self.points_input_indices = np.argwhere(np.ones((self.big_res, self.big_res, self.big_res)))
        for res in [self.small_res, self.big_res]:
            self.info[res] = {}
            self.info[res]["output_resolution"] = 16 * res
            self.info[res]["block_size"] = self.info[res]["output_resolution"] / amount_of_blocks
            self.info[res]["amount_of_points"] = int(2 ** np.ceil(np.log2(res ** 3)))

            self.info[res]["batch_size"] = (16 * 2048) // self.info[res]["amount_of_points"]
            settings_manager.data["Training"]["batch_size"] = self.info[res]["batch_size"]
            settings_manager.data["Training"]["point_amount"] = self.info[res]["amount_of_points"]
            self.info[res]["inited"] = False

            self.trainer[res] = Trainer(settings_manager)
            self.load_points_for_res(res)
        print("Loaded implicit compression models")
        self._online = online
        if online:
            super().__init__(server_port=1863, server_ip=socket.gethostname())
            self.used_target_function = ImplicitTsdfDecoderServer.on_new_client
            print("Started server")

    def load_points_for_res(self, used_res: int):
        amount_of_points = self.info[used_res]["amount_of_points"]
        points = ImplicitTsdfDecoderServer.get_grid_points(used_res, amount_of_points)
        batch_size = (16 * 2048) // amount_of_points
        points = np.tile(points.reshape((-1, *points.shape)), (batch_size, 1, 1))

        # set up the data for the prediction step
        tsdf_empty = np.zeros((batch_size, amount_of_points, 1))
        class_on_hot = np.zeros((batch_size, amount_of_points, self.number_of_classes))
        # shape = batch_size, point_amount, coords_size + output_size + number_of_classes
        coord_combined = np.concatenate([tsdf_empty, class_on_hot, points], axis=2)
        coord_combined = tf.convert_to_tensor(coord_combined.astype(np.float32))
        if used_res in self.info:
            c_info = self.info[used_res]
            c_trainer = self.trainer[used_res]
            if not c_info["inited"]:
                # set the used ids so that each latent value is used
                c_trainer.used_ids_var.assign(tf.convert_to_tensor(np.reshape(np.arange(0, batch_size, 1), (batch_size, 1)).astype(np.int32)))
            c_trainer.coord_var.assign(coord_combined)
            c_trainer.update_current_inputs()
            if not c_info["inited"]:
                # this is only done to init the weights so that the loading works
                _, _, _, _, input_to_gen = c_trainer.get_current_inputs_live()
                _ = c_trainer.gen(input_to_gen, training=False).numpy()
                c_trainer.load_weights(str(self.weights_path))
                self.info[used_res]["inited"] = True
        else:
            raise RuntimeError(f"Unknown res: {used_res}")

    def get_predictions_for_batch_of_indices(self, latent_block: np.ndarray, indices: np.ndarray, used_res: int):
        batch_size = self.info[used_res]["batch_size"]
        if indices.shape[0] < batch_size:
            indices = np.concatenate((indices, np.zeros((batch_size - indices.shape[0], 3), dtype=indices.dtype)), axis=0)
        used_latent_values = latent_block[indices[:, 0], indices[:, 1], indices[:, 2]]
        conv_input_latent_vec = tf.convert_to_tensor(used_latent_values)
        c_trainer = self.trainer[used_res]
        c_trainer.latent_variable.assign(conv_input_latent_vec)
        predictions = c_trainer.predict_current_inputs_live().numpy()
        tsdf_predictions = predictions[:, 0]
        class_predictions = np.argmax(predictions[:, 1:], axis=-1).astype(np.uint8)
        return tsdf_predictions.reshape(batch_size, -1), class_predictions.reshape(batch_size, -1)

    def create_tsdf_block(self, latent_block: np.ndarray):
        latent_block = latent_block.astype(np.float32)
        latent_threshold = 5e-2
        might_be_empty = np.mean(np.abs(latent_block - self.empty_latent_vec), axis=-1) < latent_threshold
        might_be_filled = np.mean(np.abs(latent_block - self.filled_latent_vec), axis=-1) < latent_threshold
        might_have_surface = np.logical_not(np.logical_or(might_be_filled, might_be_empty))
        print(f"Might have surface: {np.sum(might_have_surface)}")

        # start with everything free
        output_block = np.ones((self.output_resolution, self.output_resolution, self.output_resolution)) * self.truncation_threshold
        class_block = np.zeros(output_block.shape, dtype=np.uint8)
        if np.sum(might_be_filled) > 0:
            # change the already filled ones
            repeated_filled_blocks = might_be_filled
            for i in range(3):
                repeated_filled_blocks = np.repeat(repeated_filled_blocks, repeats=self.big_res, axis=i)
            output_block[repeated_filled_blocks] = -self.truncation_threshold

        t = time.time()
        might_have_surface_indices = np.argwhere(might_have_surface)
        small_batch_size = self.info[self.small_res]["batch_size"]
        surface_indices = []
        amount_of_indices = len(might_have_surface_indices)
        for used_indices in range(0, amount_of_indices, small_batch_size):
            indices = might_have_surface_indices[used_indices: used_indices + small_batch_size]
            max_amount = min(amount_of_indices - used_indices, small_batch_size)
            tsdf_predictions, class_predictions = self.get_predictions_for_batch_of_indices(latent_block, indices, self.small_res)
            occluded_areas = np.any(tsdf_predictions <= 0, axis=1)
            for index, indice in enumerate(indices[occluded_areas[:max_amount]]):
                current_indices = self.points_input_indices + (indice * self.big_res)
                output_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = -self.truncation_threshold
                #class_selected = np.unique(class_predictions[index], return_counts=True)
                #class_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = class_selected[0][np.argmax(class_selected[1])]
            free_areas = np.any(tsdf_predictions > 0, axis=1)
            has_surface = np.logical_and(occluded_areas, free_areas)
            surface_indices.extend(list(indices[has_surface[:max_amount]]))
        surface_indices = np.array(surface_indices)
        print(f"Small prediction: {time.time() - t:0.4f}s, left: {surface_indices.shape[0]}")

        t = time.time()
        a = []
        amount_of_indices = surface_indices.shape[0]
        big_batch_size = self.info[self.big_res]["batch_size"]
        for used_indices in range(0, amount_of_indices, big_batch_size):
            indices = surface_indices[used_indices: used_indices + big_batch_size]
            tsdf_predictions, class_predictions = self.get_predictions_for_batch_of_indices(latent_block, indices, self.big_res)
            max_amount = min(amount_of_indices - used_indices, small_batch_size)
            a_s = time.time()
            for indice, tsdf_prediction, class_prediction in zip(indices, tsdf_predictions, class_predictions):
                current_indices = self.points_input_indices + (indice * self.big_res)
                output_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = tsdf_prediction
                class_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = class_prediction
            a.append(time.time() - a_s)
        print(f"Done predicting: {time.time() - t:.4f}s")
        return output_block, class_block.astype(np.uint8)

    @staticmethod
    def get_grid_points(block_resolution: int, amount_of_requested_points: int) -> np.ndarray:
        # the border add should not be bigger than 10% = 1.0 to 1.1
        if block_resolution ** 3 == amount_of_requested_points:
            block_resolution = int(block_resolution)
            points_input = np.argwhere(np.ones((block_resolution, block_resolution, block_resolution))).astype(np.float32)
            points_input /= float(block_resolution) * 0.5
            points_input -= 1
            points_input += 1 / float(block_resolution)
        else:
            points_input = np.random.uniform(-0.99, 0.99, (amount_of_requested_points, 3))
        return points_input

    def on_new_client(self, connection: socket.socket, addr):
        """
        For each new client this function gets called it receives the data from the client and sends the answer back.

        This general server can only respond with a mirroring of the content

        :param connection: Open connection to the client
        :param addr: Addr info from the open socket call
        """
        data = self._receive_data(connection)
        if data is not None:
            if isinstance(data, EncodedSceneMessage):
                voxel, class_output = self.create_tsdf_block(data.content)
                msg = SceneMessage(voxel, class_output, self.truncation_threshold)
                s = time.time()
                self._send_data(connection, msg)
                print(time.time() - s)
            else:
                self._send_data(connection, "unknown command")
        self.close_connection(connection)

@numba.njit
def fuse_output_together(indices, tsdf_predictions, class_predictions, used_output_block, used_class_block, batch_size, big_res, points_input_indices):
    for index in range(batch_size):
        current_indices = points_input_indices + (indices[index] * big_res)
        used_output_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = tsdf_predictions[index]
        used_class_block[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] = class_predictions[index]


if __name__ == "__main__":

    implicit_tsdf_decoder_server = ImplicitTsdfDecoderServer()
    implicit_tsdf_decoder_server.run()
