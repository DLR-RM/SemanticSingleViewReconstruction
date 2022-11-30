# Data generation

This README explains how the data for the training of the three different networks was created.

We warn that this is extremely computational demanding. 
We spend around 15.000 GPU hours on the generation of the dataset.
The scripts provided here only allow the computation on one machine, we removed any system to generate these on a cluster as this would require cluster specific knowledge.
One machine this method could take up to two years, be aware that our dataset was over 15 TB in the end.

The code for these generated data parts is present:

* color images
* surface normal images
* TSDF point cloud per image
* compressed point cloud per image
  * boundary blocks
  * non-boundary blocks
* loss map for the compressed scene 
* tf records for all three networks


## Color and surface normal images

At first, we need to create some color and surface normal images. For this, we rely on the 3D FRONT and 3D FUTURE dataset.
You can download them [Here](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset). 

Be aware that we obtained our 3D FRONT version 2021, they have changed the dataset before without changing the version number.

After downloading this, we also need [cctextures](https://ambientcg.com/), you can use the download script provided inside BlenderProc for this. 

```shell script
pip install blenderproc
blenderproc download cc_textures ${cc_textures_output}
```

Activate the environment and run the render script, check [here](../svr/README.md) if you haven't installed the conda environment.

```shell script
conda activate SemanticSVR
```

You can then start the rendering, you need the three folders from the 3D FRONT and 3D FUTURE dataset: `${3D-FRONT-JSON}`, `${3D-FUTURE-model}`, and `${3D-FRONT-texture}`.
And you need the `${cc_textures_output}` folder, downloaded above.

```shell script
python data_generation/blenderproc/render_all_scenes.py ${3D-FRONT-JSON} ${3D-FUTURE-model} ${3D-FRONT-texture} data/images data/front_mesh ${cc_textures_output}
```

This generation script will try to place 25 cameras per scene, and for each placed camera we render the color, depth, surface normal, and semantic segmentation.
For each camera pose, we also generate three color renderes with replaced textures from the cctexture dataset. 
These will be used for the training of the U-Net to ensure that it generalizes well.

## TSDF point cloud

To generate the TSDF point cloud head over to the [SDFgen page](SDFGen/README.md), after your done you can continue in this README.md.

```shell script
python data_generation/SDFGen/generate_sdf_files.py data/front_mesh data/tsdf_point_cloud
```

## Compressing TSDF point clouds

For the compression of the generated TSDF point clouds, head over to [tsdf compression page](tsdf_compression/README.md).

## Loss Map for the compressed scene

Head over to the [Loss Calculator page](LossCalculator/README.md).

## TF Records

After all files have been generated the tf records for the training can be generated, head over to the [TF Record Page](tf_record_generation/README.md).
