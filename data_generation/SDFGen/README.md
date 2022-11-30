# TSDF Point Cloud Generation

## Building the SDFGen tool

Before building the project you have to download the `hdf5-1.10.6` package: [Here](https://www.hdfgroup.org/downloads/hdf5/).
Then change the path in the `CMakeLists.txt`.

This process has to be repeated for TClap: `tclap-1.2.2`: [Here](https://tclap.sourceforge.net/v1.2/index.html)
After changing both paths you can build the SDFGen tool with the following commands:

```shell script
cd data_generation/SDFGen
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

## Creating the TSDF point clouds

For each color image produced in the last step, we now want to produce a TSDF point cloud.

```shell script
conda activate SemanticSVR
python data_generation/SDFGen/generate_sdf_files.py data/front_mesh data/tsdf_point_cloud
```

This will generate for each generated image a TSDF point cloud with 2.000.000 points. 
However, it will fail if too many of the visible normals point outwards, indicating that some of the visible objects are broken.
As we do not provide a filtering on the used objects it might be that more objects are rejected than in our experiments.
This could be fixed my manually removing the objects, which have broken surface normals. 


