import datetime
import logging
import sys
import os
import socket
import shutil
from pathlib import Path

import tensorflow as tf
import yaml

class NotFoundException(Exception):
    pass


class SettingsManager(object):
    current_time = None

    def __init__(self, model_file_path: Path = None, app_file_path: Path = None, start_logging=True):
        if model_file_path is None:
            model_file_path = Path(__file__).parent.parent.parent / "settings" / "settings.yaml"
        if app_file_path is None:
            app_file_path = Path(__file__).parent.parent.parent / "settings" / "app_config.yaml"
        model_file_path, app_file_path = Path(model_file_path), Path(app_file_path)

        if not model_file_path.exists():
            raise Exception(f"The file path to the config file is invalid: {model_file_path}")
        if not app_file_path.exists():
            raise Exception(f"The file path to the config file is invalid: {app_file_path}")

        with open(model_file_path, "r") as file:
            self.data = yaml.Loader(file).get_data()

        with open(app_file_path, "r") as file:
            self.app_data = yaml.Loader(file).get_data()
        if SettingsManager.current_time is None:
            SettingsManager.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

        self.base_log_dir = Path(__file__).parent.parent.parent / "logs" / SettingsManager.current_time

        if start_logging:
            self.start_logger()

    def start_logger(self):
        if not self.base_log_dir.exists():
            os.makedirs(self.base_log_dir)
        logging.basicConfig(filename=self.base_log_dir / "log.txt", level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s: %(message)s', "%H:%M:%S")
        handler.setFormatter(formatter)
        root.addHandler(handler)

        logging.info(f"This test was started at: {SettingsManager.current_time}")
        self.train_log_dir = self.base_log_dir / 'train'
        self.validation_log_dir = self.base_log_dir / 'validation'
        self.train_summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))
        self.val_summary_writer = tf.summary.create_file_writer(str(self.validation_log_dir))

        self.copy_current_status_of_project()

    def get_keys(self, used_data=None):
        if isinstance(used_data, str) and used_data.lower() == "model_data":
            return list(self.data.keys())
        elif isinstance(used_data, str) and used_data.lower() == "app_data":
            return list(self.app_data.keys())
        else:
            final_keys = list(self.data.keys())
            final_keys.extend(list(self.app_data.keys()))
            return list(set(final_keys))

    def set_element(self, used_total_key, new_value, used_split="/"):
        def search_for_key(x, total_key, split):
            y = x
            for key in total_key.split(split):
                if isinstance(x, dict) and (key in x or "rmc" in x and "home" in x):
                    if key in x:
                        y = x[key]
                    else:
                        # filter the current key
                        current_platform = "rmc" if socket.gethostname().startswith("rmc-") else "home"
                        y = x[current_platform][key]
                elif isinstance(x, list) and int(key) < len(x):
                    y = x[int(key)]
                elif isinstance(x, dict):
                    raise NotFoundException(f"This key: {key} is not inside of the current dictionary: {x}")
                else:
                    raise NotFoundException(f"This key: {key} is not inside of this: {x}")
                if isinstance(y, dict):
                    x = y
                    break
            y[total_key.split(split)[-1]] = new_value

        try:
            search_for_key(self.data, used_total_key, used_split)
        except NotFoundException as e:
            try:
                search_for_key(self.app_data, used_total_key, used_split)
            except NotFoundException as e2:
                raise e

    def __call__(self, used_total_key, used_split="/", used_data=None):
        def search_for_key(x, total_key, split):
            for key in total_key.split(split):
                if isinstance(x, dict) and (key in x or "rmc" in x and "home" in x):
                    if key in x:
                        x = x[key]
                    else:
                        # filter the current key
                        current_platform = "rmc" if socket.gethostname().startswith("rmc-") else "home"
                        x = x[current_platform][key]
                elif isinstance(x, list) and int(key) < len(x):
                    x = x[int(key)]
                elif isinstance(x, dict):
                    raise NotFoundException(f"This key: {key}:{used_total_key} is not inside of the current dictionary: {x}")
                else:
                    raise NotFoundException(f"This key: {key}:{used_total_key} is not inside of this: {x}")

            # filter the current key
            if isinstance(x, dict) and ("rmc" in x and "home" in x):
                current_platform = "rmc" if socket.gethostname().startswith("rmc-") else "home"
                x = x[current_platform]
            x = self._check_type(x)
            return x

        if isinstance(used_data, str) and used_data.lower() == "app_data":
            x = search_for_key(self.app_data, used_total_key, used_split)
        elif isinstance(used_data, str) and used_data.lower() == "model_data":
            x = search_for_key(self.data, used_total_key, used_split)
        else:
            try:
                x = search_for_key(self.data, used_total_key, used_split)
            except NotFoundException as e:
                try:
                    x = search_for_key(self.app_data, used_total_key, used_split)
                except NotFoundException as e2:
                    raise e
        return x

    def _check_type(self, x):
        if isinstance(x, str):
            if x.lower() == "true":
                return True
            elif x.lower() == "false":
                return False
            elif "e-" in x or "e+" in x:
                try:
                    y = float(x)
                    return y
                except ValueError as e:
                    return x
        return x

    def copy_current_status_of_project(self):
        base_folder = Path(__file__).parent.parent
        if base_folder.name != "scene_reconstruction":
            raise Exception(f"Something weird happened, base folder name is incorrect: {base_folder}")
        goal_path = self.base_log_dir / "code"
        if not goal_path.exists():
            os.makedirs(goal_path)
        for type_ending in ["py", "yaml", "yml", "sh", "md"]:
            for type_file in base_folder.rglob(f"*.{type_ending}"):
                if "logs" in str(type_file).split("/"):
                    continue
                goal_type_file = Path(str(type_file.absolute()).replace(str(base_folder.absolute()), str(goal_path.absolute())))
                if not goal_type_file.parent.exists():
                    os.makedirs(goal_type_file.parent)
                shutil.copyfile(type_file, goal_type_file)

