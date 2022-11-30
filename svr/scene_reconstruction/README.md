# Scene Reconstruction Network

This network converts a color and surface normal image into a compressed scene representation.

## Setup

The general setup for the conda environment described on the [last page](../README.md) has to be followed.

## Running the scene reconstruction server

It is also possible to run the scene reconstruction server on its own.
This mean you need to feed a color *and* a surface normal image to it.
If only a color image is provided, an automatic request to the `normal_reconstruction_server` is made.
Make sure it is running prior to starting the client [check here](../u_net_normal/README.md).
If you only want to test this method on simple color images, see [here](../README.md).

```bash
conda activate SemanticSVR
python svr/scene_reconstruction/scene_reconstruction_server.py
```

This server starts on port `1782`.

You can now start a client in a new terminal:

```bash
conda activate SemanticSVR
python svr/scene_reconstruction/scene_reconstruction_client.py data/images/$TEST_FILE output_test.hdf5
```
 
This output `output_test.hdf5` can not be decoded with the [implicit tsdf decoder](../implicit_tsdf_decoder/README.md).

## Train your own network

If you want to train your own network, you first need to generate your own data. 
For this checkout the [data generation page](../../data_generation/README.md).
After you have generated the `tf records` for the scene reconstruction task you need to compile two C++ functions we use for optimal speed.

```bash
conda activate SemanticSVR
```

Before you can compile this you first need to add the protobuf include to your environment. 
This was the only way we found to make this work.

```bash
cd /tmp
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout 3.18.x  # must be the same version as: protoc --version
cd src
cp -r google $(dirname "${CONDA_EXE}")/../envs/SemanticSVR/lib/python3.9/site-packages/tensorflow/include/google
cd /tmp
rm -r protobuf
```

After this the build should work:

```bash 
cd svr/scene_reconstruction/loss_converter
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared loss_converter_class.cc -o loss_converter.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++14 -shared reachable_space_detector.cc -o reachable_space_detector.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

After this you can start the training with the following command:

```bash
conda activate SemanticSVR
python svr/scene_reconstruction/train.py data/scene_reconstruction_tf_records scene_rec_logs -m 8640
```

The training takes around 18 GB of VRAM.
