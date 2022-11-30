import logging

import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D 

from svr.scene_reconstruction.model.layers.layer_interface import LayerInterface
from svr.scene_reconstruction.model.layers.inception_layer import InceptionLayer
from svr.scene_reconstruction.model.layers.split_layer import SplitLayer

class TreeLayer(LayerInterface):

    def __init__(self, name: str):
        super(TreeLayer, self).__init__(name=name)
        self.height: int = LayerInterface.settings("TreeModel/height")
        self._amount_of_filters = LayerInterface.settings("TreeModel/filters_for_level")
        self._amount_of_res_blocks = LayerInterface.settings("TreeModel/amount_of_res_block_per_level")
        self._input_structure = LayerInterface.settings("TreeModel/input_structure")
        self._result_resolution = LayerInterface.settings("TreeModel/result_resolution")
        self._amount_of_filters_in_first_3d_layer = LayerInterface.settings("TreeModel/amount_of_filters_in_first_3D_layer")
        self._desired_amount_of_filters_in_last_2d_layer = self._amount_of_filters_in_first_3d_layer // int(2**self.height) * self._result_resolution
        self._reduce_levels = []
        for filter_amount in self._input_structure:
            if filter_amount != -1:
                self._reduce_levels.append(InceptionLayer(filter_amount, name="ReduceLevel"))
            else:
                self._reduce_levels.append(MaxPooling2D())

        self._pre_levels = [[] for _ in range(self.height)]
        self._post_levels = [[] for _ in range(self.height)]
        self._levels = [[] for _ in range(self.height)]
        last_amount_of_filters = 6 if LayerInterface.settings("Training/use_normal") else 3
        for level_id in range(self.height):
            current_level_amount_of_filters = self._amount_of_filters[level_id]
            current_amount_of_res_blocks = self._amount_of_res_blocks[level_id]

            amount_of_nodes_in_layer = 2**level_id

            for node_id in range(amount_of_nodes_in_layer):
                current_name = f"level_{level_id}_{node_id}"
                info = "\t" * level_id + f"Id: {level_id}, node: {node_id},"
                if current_level_amount_of_filters != last_amount_of_filters:
                    # change the amount of filters
                    self._pre_levels[level_id].append(InceptionLayer(current_level_amount_of_filters, filter_size=1, name=f"Pre_InceptionLayer_{node_id}_{level_id}_before"))
                    info += f" {last_amount_of_filters} filters to"
                info += f" filters: {current_level_amount_of_filters}, res block: {current_amount_of_res_blocks}"
                self._levels[level_id].append(SplitLayer(current_level_amount_of_filters,
                                                         current_amount_of_res_blocks, name=current_name))

                if level_id == self.height - 1 and current_level_amount_of_filters != self._desired_amount_of_filters_in_last_2d_layer:
                    first_part = InceptionLayer(self._desired_amount_of_filters_in_last_2d_layer, filter_size=1, name=f"Post_InceptionLayer_{node_id}_{level_id}_front")
                    second_part = InceptionLayer(self._desired_amount_of_filters_in_last_2d_layer, filter_size=1, name=f"Post_InceptionLayer_{node_id}_{level_id}_back")
                    self._post_levels[level_id].append((first_part, second_part))
                    info += f" to {last_amount_of_filters} filters"
                logging.info(info)
            last_amount_of_filters = current_level_amount_of_filters

        self._concat_layers = [tf.keras.layers.Concatenate(axis=-1, name=f"Concat_part_{i}") for i in range(self._amount_of_filters_in_first_3d_layer)]
        self._last_concat = tf.keras.layers.Concatenate(axis=-1, name="Concat_last")

    def call(self, input_tensor, training=False):
        x = input_tensor
        for reduce_level in self._reduce_levels:
            x = reduce_level(x)

        last_layer_2d_result = self._recursive_depth_first_tree_parsing(x, 0, 0)
        combined_output = self.combine_2d_layers_to_3d(last_layer_2d_result)
        return combined_output

    def _recursive_depth_first_tree_parsing(self, input_tensor, node_index: int, level: int):
        if level == self.height:
            return [input_tensor]
        pre_layers, layers, post_layers = self._pre_levels[level], self._levels[level], self._post_levels[level]
        x = input_tensor
        if pre_layers:
            x = pre_layers[node_index](x)
        first_output, second_output = layers[node_index](x)
        if post_layers:
            first_output = post_layers[node_index][0](first_output)
            second_output = post_layers[node_index][1](second_output)
        ret = self._recursive_depth_first_tree_parsing(first_output, node_index * 2, level + 1)
        ret.extend(self._recursive_depth_first_tree_parsing(second_output, node_index * 2 + 1, level + 1))
        return ret

    def combine_2d_layers_to_3d(self, inputs):
        individual_size = int(self._result_resolution // len(inputs))
        amount_of_new_channels = self._amount_of_filters_in_first_3d_layer
        multi_dim_collection = []
        for i in range(amount_of_new_channels):
            current_collection = []
            current_start = i * individual_size
            current_end = (i + 1) * individual_size
            for element in inputs:
                current_collection.append(element[:, :, :, current_start:current_end])
            new_3d_channel = self._concat_layers[i](current_collection)
            new_3d_channel = tf.expand_dims(new_3d_channel, axis=-1)
            multi_dim_collection.append(new_3d_channel)
        return self._last_concat(multi_dim_collection)








