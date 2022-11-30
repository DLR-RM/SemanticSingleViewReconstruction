import os
import pickle
import time
import shutil
from typing import Tuple
from pathlib import Path
import numpy as np


from svr.implicit_tsdf_decoder.model.similarity_manager import load_data, calculate_similarity_vec, calculate_similarity_vec_non_boundary
from svr.implicit_tsdf_decoder.model.convert_tsdf_to_blocked_tsdf_file import convert_tsdf_to_blocked_tsdf_file_path
from svr.implicit_tsdf_decoder.model.filter_classes import FilterClasses


class SimilaritySetter(object):

    def __init__(self, data_base_folder: Path, only_non_boundary: bool = False):

        start_time = time.time()
        self._only_non_boundary = only_non_boundary
        self.database_folder = Path(data_base_folder)
        if not only_non_boundary:
            data_base_internal_path = self.database_folder / "boundary.pickle"
        else:
            data_base_internal_path = self.database_folder / "non_boundary.pickle"

        with open(data_base_internal_path, "rb") as file:
            tree, giant_similarity_vector, giant_latent_vector = pickle.load(file)
        print(f"Took: {time.time() - start_time}s to load the database.pickle file for the similarity comparison.")

        self.tree = tree
        self.similarity_vector = giant_similarity_vector 
        self.latent_vector = giant_latent_vector

    def find_most_similar_for_vector(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = self.tree.query(vector, k=1)[1]
        return self.latent_vector[index] #, self.giant_latent_weight_1[index], self.giant_latent_weight_2[index]

    def find_most_similar(self, point: np.ndarray, dist: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._only_non_boundary:
            check_vec = calculate_similarity_vec(point, dist, class_ids)
        else:
            check_vec = calculate_similarity_vec_non_boundary(point, dist, class_ids)
        return self.find_most_similar_for_vector(check_vec)
