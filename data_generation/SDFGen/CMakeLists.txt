cmake_minimum_required(VERSION 3.5)
project(sdfgen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -lstdc++ -lm")
##set(CMAKE_VERBOSE_MAKEFILE ON)
##SET(CMAKE_CXX_FLAGS "-march=native -ffast-math -O6 -S")

include_directories(./include)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -lstdc++ -lm")

set (LIB_TYPE STATIC)
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

set(HDF5_DIR /home/max/3rdparty/hdf5-1.10.6/HDF_Group/HDF5/1.10.6/share/cmake/hdf5)

find_package (HDF5 NAMES hdf5 COMPONENTS C ${SEARCH_TYPE})
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
set (LINK_LIBS ${LINK_LIBS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

include_directories(/home/max/3rdparty/tclap-1.2.2/include)
add_compile_options(-pthread)

SET(CMAKE_EXE_LINKER_FLAGS "-pthread")
add_executable(sdfgen main.cpp reader/SuncgReader.cpp reader/SuncgReader.h reader/ObjReader.cpp reader/ObjReader.h util/Utility.h util/StopWatch.h util/StopWatch.cc geom/Line.h geom/Line.cc geom/math/Point.h geom/math/Point_i.h geom/Polygon.cpp geom/Polygon.h geom/math/DistPoint.h container/Array3D.h container/Array3D_i.h geom/BoundingBox.cpp geom/BoundingBox.h container/Space.cpp container/Space.h test/PolygonTest.h test/PolygonTest.cc geom/math/AvgNumber.h container/Octree.cpp container/Octree.h geom/PolygonCubeIntersection.h geom/PolygonCubeIntersection.cpp util/Hdf5Writer.h util/CSVReader.h util/CSVReader.cpp geom/Cube.h util/ObjWriter.cpp util/ObjWriter.h util/StopWatchPrinter.cpp util/StopWatchPrinter.h geom/Beam.cpp geom/Beam.h container/PointOctree.cpp container/PointOctree.h geom/math/deform.h reader/Front3DReader.cpp reader/Front3DReader.h reader/PolygonReader.h util/CamPose.h util/MathUtilityFunctions.h util/OldConvertCameraPosesFunctions.h util/NormalCheckerRayCaster.cpp util/NormalCheckerRayCaster.h reader/ReplicaReader.h)


target_compile_options(sdfgen PRIVATE -march=native -ffast-math -lm)
target_link_libraries(sdfgen PRIVATE ${HDF5_C_LIBRARIES})
target_link_libraries (sdfgen ${LINK_LIBS})
