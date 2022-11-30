import datetime
import logging
import sys
import os
import socket
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class NotFoundException(Exception):
    pass


class SettingsManager(object):
    current_time = None

    def __init__(self, model_file_path: Path, app_file_path: Path, start_logging=True):
        if not os.path.exists(model_file_path):
            raise Exception(f"The file path to the config file is invalid: {model_file_path}")
        if not os.path.exists(app_file_path):
            raise Exception(f"The file path to the config file is invalid: {app_file_path}")

        with open(model_file_path, "r") as file:
            self.data = yaml.Loader(file).get_data()

        with open(app_file_path, "r") as file:
            self.app_data = yaml.Loader(file).get_data()
        if SettingsManager.current_time is None:
            SettingsManager.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_folder = os.path.abspath(os.path.dirname(__file__))
        self.base_log_dir = os.path.abspath(os.path.join(current_folder, '../..', 'logs', SettingsManager.current_time))

        if start_logging:
            self.start_logger()

    def start_logger(self):
        if not os.path.exists(self.base_log_dir):
            os.makedirs(self.base_log_dir)
        logging.basicConfig(filename=os.path.join(self.base_log_dir, "log.txt"), level=logging.DEBUG,
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
        self.latent_log_dir = os.path.join(self.base_log_dir, 'train', 'latent')
        self.train_log_dir = os.path.join(self.base_log_dir, 'train', 'train')
        self.validation_log_dir = os.path.join(self.base_log_dir, 'validation', 'latent')
        self.validation_mean_log_dir = os.path.join(self.base_log_dir, 'validation_latent_mean')

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


class YParams(object):

    def __init__(self, settings_manager: SettingsManager, used_data=None):
        self.current_hparams = {}
        c_int = lambda name, min, max: hp.HParam(name, hp.IntInterval(min, max))
        c_bool = lambda name: hp.HParam(name, hp.IntInterval(0, 1))
        c_float = lambda name, min, max: hp.HParam(name, hp.RealInterval(min, max))

        def c_int_a(name, min, max, amount):
            ret = [c_int(name + "/" + str(index), min, max) for index in range(amount)]
            ret.append(c_int(name + "/amount", 1, amount))
            return ret

        self.hparams = [c_int("Training/summary_steps", 0, 5000), c_int("Training/latent_steps", 0, 20000),
                        c_int("Training/gen_steps", 0, 20000), c_int("Training/amount_of_validation_checks", 0, 1000),
                        c_int("Training/batch_size", 1, 1024), c_int("Training/point_amount", 1, 65536),
                        c_float("Training/latent_learning_rate", 1e-10, 5e-1),
                        c_float("Training/latent_learning_drop_decay", 1e-10, 2.0),
                        c_int("Training/latent_learning_drop_epoch_amount", 1, 15000),
                        hp.HParam("Training/latent_learning_rate_mode", hp.Discrete(["STEP_DECAY", "FIXED"])),
                        c_float("Training/gen_learning_rate", 1e-10, 5e-1), c_int("Generator/latent_dim", 1, 16384),
                        c_int("Generator/coords_size", 1, 4), c_int("Generator/output_size", 1, 128),
                        c_bool("Generator/use_batch_norm"), c_float("DataLoader/org_trunc_threshold", 0.0, 5.0),
                        c_float("Generator/trunc_threshold", 0.0, 5.0),
                        c_int("DataLoader/amount_of_blocks_per_voxel", 0, 1000),
                        c_bool("DataLoader/load_only_blocks_with_boundary"),
                        c_float("DataLoader/boundary_selection_scale", 0.1, 2.0),
                        c_float("DataLoader/min_point_amount", 0.01, 1.0),
                        c_float("DataLoader/tsdf_min_threshold_to_use_block", 0.0001, 10.0),
                        c_int("DataLoader/resolution", 1, 128), c_bool("LossManager/add_surface_weights"),
                        c_float("LossManager/surface_loss_weight", 1e-8, 1000.0),
                        hp.HParam("LossManager/surface_loss_type", hp.Discrete(["GAUSS", "EXP"])),
                        c_bool("LossManager/add_sign_weights"), c_bool("LossManager/add_corner_weights"),
                        c_int("DataLoader/validation_size", 0, 100000), c_int("DataLoader/shuffle_size", 0, 100000000),
                        c_bool("Generator/fourier_use_mapping"), c_float("Generator/fourier_mapping_scale", 0.0, 10000.0),
                        c_int("Generator/fourier_mapping_size", 1, 16384),
                        hp.HParam("Generator/activation_type", hp.Discrete(["RELU", "SIREN"])),
                        c_float("Generator/sinus_w0_first", 1e-7, 1000000.0),
                        c_float("Generator/sinus_w0_hidden", 1e-8, 1000000.0),
                        c_bool("Training/use_gradient_smoothing"), c_float("Training/gradient_size", 1e-8, 10000.0),
                        c_float("Training/gradient_loss_scaling", 1e-7, 1000000.0),
                        c_bool("DataLoader/select_one_batch_per_cube"),
                        c_bool("Generator/use_classes"),
                        c_int("Generator/number_of_classes", 0, 100),
                        c_float("Generator/class_weighting_loss", 0.0, 10000.0),
                        hp.HParam("DataLoader/mode", hp.Discrete(["normal", "single_file"]))]
        self.hparams.extend(c_int_a("Generator/layers", 1, 4096, 7))
        self.hparams.extend(c_int_a("Generator/concats", 0, 1, 7))
        self.hparams.extend(c_int_a("Generator/final_class_layers", 0, 4096, 7))

        self._hparams_names = [ele._name for ele in self.hparams]

        for key in settings_manager.get_keys(used_data):
            for second_key in settings_manager(key, used_data=used_data).keys():
                data = settings_manager(key + "/" + second_key, used_data=used_data)
                def _append(name, d):
                    if "folder_path" in name or "file_path" in name:
                        return
                    if name not in self._hparams_names:
                        raise Exception(f"The {name} is unknown, add it to the hparam list!")
                    relevant_param = [ele for ele in self.hparams if ele._name == name][0]
                    if isinstance(d, float) or isinstance(d, int) or isinstance(d, np.int) or isinstance(d, np.float):
                        if not (relevant_param.domain.min_value <= d <= relevant_param.domain.max_value):
                            raise Exception(f"This value {d} is not inside of the domain: {relevant_param.domain}")
                    elif isinstance(relevant_param.domain, hp.Discrete):
                        if d not in relevant_param.domain.values:
                            raise Exception(f"This value {d} is not in the domain {relevant_param.domain}!")
                    else:
                        raise Exception(f"This type is not usable: {d}, {type(d)} for {relevant_param.domain}!")
                    self.current_hparams[relevant_param] = d

                if isinstance(data, list):
                    for index, ele in enumerate(data):
                        _append(key + "/" + second_key + "/" + str(index), ele)
                    _append(key + "/" + second_key + "/amount", len(data))
                else:
                    _append(key + "/" + second_key, data)

    def get_hparams(self):
        return self.current_hparams
        ret = {}
        for hparam in self.hparams:
            for value in hparam.domain.values:
                ret[hparam] = value
                break
        return ret


class ModelParams(YParams):

    def __init__(self, settings_manager: SettingsManager):
        super(ModelParams, self).__init__(settings_manager, used_data="model_data")


class AppConfig(YParams):

    def __init__(self, settings_manager: SettingsManager):
        super(AppConfig, self).__init__(settings_manager, used_data="app_data")


def limit_gpu_usage():
    gpus = tf.config.list_physical_devices('GPU')
    # Currently, memory growth needs to be the same across GPUs
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
