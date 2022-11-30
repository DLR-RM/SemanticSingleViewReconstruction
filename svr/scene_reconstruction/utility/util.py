import tensorflow as tf


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
