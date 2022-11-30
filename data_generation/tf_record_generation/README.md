# TF Records for the training

## U-Net Surface Normals

At first, we create the tf records for the U-Net.

```shell script
conda activate SemanticSVR
python data_generation/tf_record_generation/tf_records_for_surface_normals.py data/images data/surface_normal_tf_records
```

This will use all the color images, even the color images with switched textures and map them to the correct normal images.

## TSDF Point Compression

The generation for the implicit TSDF point cloud network is similarly easy:

```shell script
conda activate SemanticSVR
python data_generation/tf_record_generation/tf_records_for_tsdf_compression.py data/tsdf_point_cloud data/tsdf_point_cloud_tf_records
```

## Scene Reconstruction

For the scene reconstruction, the compressed scene, the color and normal image and the loss map are needed:
If all of these have been generated before the tf records can be generated via:

```shell script
conda activate SemanticSVR
python data_generation/tf_record_generation/tf_records_for_scene_reconstruction.py data/compressed_scene data/images data/tsdf_point_cloud data/scene_reconstruction_tf_records
```