
import numpy as np

class Message(object):
    """
    Base class of Message
    """
    def __init__(self):
        self.content = None


class ImageMessage(Message):

    def __init__(self, image: np.ndarray):
        self.content = image

class EncodedSceneMessage(Message):

    def __init__(self, scene: np.ndarray):
        self.content = scene

class SceneMessage(Message):

    def __init__(self, voxel_block: np.ndarray, class_block: np.ndarray, threshold: float):
        if voxel_block.dtype != np.uint16:
            max_value = float(np.iinfo(np.uint16).max) * 0.5
            voxel_block = (voxel_block / threshold * max_value + max_value).astype(np.uint16)

        self.content = {"voxel_block": voxel_block, "class_block": class_block.astype(np.uint8)}
