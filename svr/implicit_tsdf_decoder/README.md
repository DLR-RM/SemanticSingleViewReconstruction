
# Implicit TSDF Decoder

## Server and Client

We also provide a single server and client for decoding compressed latent representations.

You can start this server via:

```bash
conda activate SemanticSVR
python svr/implicit_tsdf_decoder/implicit_tsdf_decoder_server.py
```

This server will start on port `1863`.

In another terminal you can now query this server:

```bash
conda activate SemanticSVR
python svr/implicit_tsdf_decoder/implicit_tsdf_decoder_server.py ${INPUT_PATH} ${OUTPUT_PATH}
```

The input could be one of the generated elements in `data/compressed_scene`, if you haven't done this yet. 
Check out the [data generation page](../../data_generation/README.md). 

## Train your own network

If you want to train your own network, you first need to generate your own data.
For this checkout the [data generation page](../../data_generation/README.md).
After you have generated the `tf records` for the tsdf point clouds you can start the training with the following command:

```bash
conda activate SemanticSVR
python svr/implicit_tsdf_decoder/train.py data/tsdf_point_cloud_tf_records implicit_decoder_logs -m 360
```

This training only uses around 4GB of VRAM.