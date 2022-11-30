
# Loss Calculator

This page explains how to create the loss maps defined in the paper, to shape the loss around the surfaces in the scene reconstruction task.

Before building this project, one needs to change the paths in the `CMakeLists.txt` you can use the same ones as before for `SDFGen`.

```shell script
cd data_generation/LossCalculator
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

After the build is complete one just needs to calculate all loss maps:

```shell script
conda activate SemanticSVR
python data_generation/LossCalculator/create_all_loss_maps.py data/tsdf_point_cloud
```

The result will be stored in the `data/tsdf_point_cloud` folder.
