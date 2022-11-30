cmake_minimum_required(VERSION 3.5)
project(Blocker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -lstdc++")

# HDF5
set(HDF5_DIR /home/max/3rdparty/hdf5-1.10.6/HDF_Group/HDF5/1.10.6/share/cmake/hdf5)
set (LIB_TYPE STATIC) # or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
find_package (HDF5 NAMES hdf5 COMPONENTS C ${SEARCH_TYPE})
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
set (LINK_LIBS ${LINK_LIBS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# TCLAP
include_directories(/home/max/3rdparty/tclap-1.2.2/include)

add_compile_options(-pthread)
SET(CMAKE_EXE_LINKER_FLAGS "-pthread")

add_executable(Blocker main.cpp src/Hdf5ReaderAndWriter.h src/Blocker.cpp src/Blocker.h src/StopWatch.h src/StopWatch.cc src/Array3D.h src/Point.h src/Point_i.h)
target_compile_options(Blocker PRIVATE -march=native -ffast-math -lm)
target_link_libraries(Blocker PRIVATE ${HDF5_C_LIBRARIES})

target_link_libraries (Blocker ${LINK_LIBS})

