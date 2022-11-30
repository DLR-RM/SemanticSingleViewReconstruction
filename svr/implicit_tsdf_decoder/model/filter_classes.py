import os
from pathlib import Path
import random

import numpy as np
import yaml


class FilterClasses(object):

    def __init__(self, resources_path=None):
        self._use_front3d_mapping = False
        front_3d_path = Path(__file__).parent.parent / "resources" / "front3d_class_mapping.yaml"
        if resources_path is None:
            is_replica = os.getenv('IS_REPLICA')
            if is_replica:
                resources_path = Path(__file__).parent.parent / "resources" / "replica_class_mapping.yaml"
                self._use_front3d_mapping = True
            else:
                resources_path = front_3d_path
        self.resource_path = resources_path

        with open(self.resource_path, "r") as file:
            self.mapping = yaml.load(file, Loader=yaml.FullLoader)
        self.mapped_ids = {}
        self.old_ids_to_name = {}
        self.new_ids_to_name = {}
        self.name_to_new_ids = {}
        self.name_to_old_ids = {}
        self._list_of_special_info = []
        self._list_of_special_classes = []
        self._construct_class_mapping()
        if self._use_front3d_mapping:
            self.front_3d_filter_classes = FilterClasses(front_3d_path)
            self.replica_mapping = {}
            for replica_id, replica_name in self.new_ids_to_name.items():
                if replica_name in self.front_3d_filter_classes.name_to_new_ids:
                    self.replica_mapping[replica_id] = self.front_3d_filter_classes.name_to_new_ids[replica_name]
                else:
                    raise Exception(f"The class: {replica_name} is not in {self.front_3d_filter_classes.name_to_new_ids.keys()}")

    def get_void_category(self):
        if self._use_front3d_mapping:
            return 22
        else:
            return self.mapped_ids[0]

    def _construct_class_mapping(self):
        """
        Constructs from the given yaml file the correct mapping from the SUNCG classes to a reduced set of classes
        """
        if "mapping" in self.mapping:
            map_names_to_ids = {}
            for category_id, info in self.mapping["mapping"].items():
                map_names_to_ids[info["name"]] = int(category_id)

            def get_id_for_name(current_name):
                for category_id, info in self.mapping["mapping"].items():
                    if current_name == info["name"]:
                        if info["use_direct"]:
                            return int(category_id)
                        elif "is" in info:
                            return get_id_for_name(info["is"])
                raise Exception("The name does not appear in the file: {}".format(current_name))
            for category_id, info in self.mapping["mapping"].items():
                category_id = int(category_id)
                self.old_ids_to_name[category_id] = info["name"]
                if info["use_direct"]:
                    self.mapped_ids[category_id] = category_id
                else:
                    if "is" in info:
                        self.mapped_ids[category_id] = get_id_for_name(info["is"])
                    elif "depends" in info and info["name"] == "pillow":
                        self._list_of_special_info.append((category_id, info["depends"]))
                    else:
                        self.mapped_ids[category_id] = self.get_void_category()

            # remap the keys to the smallest possible numbers
            used_classes = set(self.mapped_ids.values())
            used_classes = dict(zip(used_classes, range(len(used_classes))))
            for old_id, new_id in used_classes.items():
                self.new_ids_to_name[new_id] = self.old_ids_to_name[old_id]
            for id, name in self.new_ids_to_name.items():
                self.name_to_new_ids[name] = id
            for id, name in self.old_ids_to_name.items():
                self.name_to_old_ids[name] = id

            for key in self.mapped_ids.keys():
                self.mapped_ids[key] = used_classes[self.mapped_ids[key]]

            new_special_list = []
            for ele in self._list_of_special_info:
                for e in ele[1]:
                    if e not in self.name_to_new_ids:
                        raise Exception(f"For {self.old_ids_to_name[ele[0]]} the depends on something which is not used: {e}")

                new_special_list.append((ele[0], [self.name_to_old_ids[e] for e in ele[1]]))
            self._list_of_special_info = new_special_list
            self._list_of_special_classes = [e[0] for e in self._list_of_special_info]

        else:
            raise Exception("There is no mapping in the given yaml file!")

    def filter_classes(self, classes: np.ndarray) -> np.ndarray:
        """
        This function removes unused classes and maps them back to better suited candiates

        :param classes: Input classes in range
        :return: classes (reduced class range)
        """

        used_classes = np.unique(classes)
        transformed_classes = np.zeros(classes.shape)
        for c in used_classes:
            if c not in self._list_of_special_classes:
                transformed_classes[classes == c] = self.mapped_ids[c]
            else:
                # check for all special info cases, mostly pillow
                for suncg_id, list_of_old_ids in self._list_of_special_info:
                    if suncg_id == c:
                        found_counter = 0
                        last_found_class = None
                        # count if only bed or sofa is there if both is then assign both to one at random
                        for new_possible_neighbour_ids in list_of_old_ids:
                            if new_possible_neighbour_ids in classes:
                                last_found_class = new_possible_neighbour_ids
                                found_counter += 1
                        if found_counter == 1:
                            picked_class = last_found_class
                        else:
                            picked_class = random.choice(list_of_old_ids)

                        transformed_classes[classes == suncg_id] = self.name_to_new_ids[self.old_ids_to_name[picked_class]]
        if self._use_front3d_mapping:
            # maps the classes from the internal representation now to the front3d mapping
            final_transformed_classes = np.zeros(classes.shape, dtype=np.uint8)
            for old_class_id, new_class_id in self.replica_mapping.items():
                final_transformed_classes[transformed_classes == old_class_id] = new_class_id
            return final_transformed_classes
        else:
            return transformed_classes
