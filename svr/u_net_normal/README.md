# U-Net for the prediction of surface normals 

This folder contains the specification of our U-Net designed to predict the surface normals for a given color image.

## Setup

The general setup for the conda environment described on the [last page](../README.md) has to be followed.

## Run surface normal prediction 

To start the server execute the following commands:

```bash
conda activate SemanticSVR
python svr/u_net_normal/normal_reconstruction_server.py
```

This script assumes that your `1536` port is currently free.  

After you started your server you can use the client script to query this server with images.
For this you can use the client script, just create a client, read in a color image and present it to the server:

```python
    client = NormalReconstructionClient()
    # read in color image
    test_img = np.asarray(Image.open(str(test_img_path)))
    # predict surface normals in range -1 to 1
    normal_img = client.get_normal_img(test_img)
```

A quick visualization of the results can be shown with:

```bash
conda activate SemanticSVR
python svr/u_net_normal/normal_reconstruction_client.py
```

## Train your own network

If you want to train your own network, you first need to generate your own data. 
For this checkout the [data generation page](../../data_generation/README.md).
After you have generated the `tf records` for the surface normals you can start the training with the following command:


```bash
conda activate SemanticSVR
python svr/u_net_normal/train.py data/surface_normal_tf_records -m 8640 u_net_logs
```

Be aware that you need at least 12 GB of VRAM to train this network.
