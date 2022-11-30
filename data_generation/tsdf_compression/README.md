# Compressing TSDF point clouds

## Compile the Blocker

Each TSDF point cloud needs to be divided into 16³ blocks. As we use the boundary points in several blocks, we wrote a small C++ program which does this for us quickly.

Before building this project, one needs to change the paths in the `CMakeLists.txt` you can use the same ones as before for `SDFGen`.

```shell script
cd data_generation/tsdf_compression/Blocker
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```


## Database Generation

We first need to generate a database of compressed tsdf blocks, these are used to speed up the compression of the rest of points.

```shell script
conda activate SemanticSVR
python data_generation/tsdf_compression/create_database.py data/tsdf_point_cloud data/tsdf_point_cloud_database --database_size 1000
```

These 1000 compressed files for boundary and non boundary blocks can take a while.
Each file takes around 15 minutes on an NVIDIA GTX 2070.

## Conversion of all files

After the database was generated and stored in `data/tsdf_point_cloud_database`, we can generate the rest of the compressed tsdf files.

```shell script
conda activate SemanticSVR
python data_generation/tsdf_compression/convert_all_files.py data/tsdf_point_cloud data/tsdf_point_cloud_database data/compressed_scene
```

This compression takes the most time of all steps. Performing this on a single machine will take more than one or two years for 90.000 scenes!







