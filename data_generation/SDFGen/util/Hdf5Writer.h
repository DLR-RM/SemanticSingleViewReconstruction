//
// Created by Maximilian Denninger on 2019-07-30.
//

#ifndef SDFGEN_HDF5WRITER_H
#define SDFGEN_HDF5WRITER_H

#include <mutex>
#include <cstring>
#include "hdf5.h"
#include "../geom/math/DistPoint.h"
#include "../container/Array3D.h"

namespace Hdf5Writer {

    static std::mutex hdf5_writer_mutex;

	static hid_t openFile(const std::string& filePath){
		return H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	}


    static void writeListToFileId(const hid_t fileId, const std::string dataset, const std::list<DistPoint>& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        std::vector<float> array;
        const auto size = listOfPoints.size();
        array.resize(size * 5);
        int counter = 0;
        for(const auto& point : listOfPoints){
            array[counter] = point[0];
            array[counter+1] = point[1];
            array[counter+2] = point[2];
            array[counter+3] = point.getDist();
            array[counter+4] = point.getClass();
            counter += 5;
        }
        hsize_t chunk[3] = {size, 5};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, 5};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
	}

    template<std::size_t amount>
    static void writeVectorToFileId(const hid_t fileId, const std::string dataset, const std::vector<std::array<double, amount> >& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }

        std::vector<float> array;
        const auto size = listOfPoints.size();
        array.resize(size * amount);
        int counter = 0;
        for(const auto& point : listOfPoints){
            for(int i = 0; i < amount; i++){
                array[counter + i] = point[i];
            }
            counter += amount;
        }
        hsize_t chunk[3] = {size, amount};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, amount};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }

    template<std::size_t amount>
    static void writeVectorToFileId(const hid_t fileId, const std::string dataset, const std::vector<std::array<unsigned char, amount> >& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }

        std::vector<unsigned char> array;
        const auto size = listOfPoints.size();
        array.resize(size * amount);
        int counter = 0;
        for(const auto& point : listOfPoints){
            for(int i = 0; i < amount; i++){
                array[counter + i] = point[i];
            }
            counter += amount;
        }
        hsize_t chunk[3] = {size, amount};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, amount};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U8LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }

    static void writeVectorToFileId(const hid_t fileId, const std::string dataset, const std::vector<DistPoint>& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        std::vector<float> array;
        const auto size = listOfPoints.size();
        array.resize(size * 5);
        int counter = 0;
        for(const auto& point : listOfPoints){
            array[counter] = point[0];
            array[counter+1] = point[1];
            array[counter+2] = point[2];
            array[counter+3] = point.getDist();
            array[counter+4] = point.getClass();
            counter += 5;
        }
        hsize_t chunk[3] = {size, 5};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, 5};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }
    template<class T>
	static void writeContainerOfPointsToFile(const hid_t fileId, const std::string dataset, const T& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        std::vector<float> array;
        const auto size = listOfPoints.size();
        array.resize(size * 3);
        int counter = 0;
        for(const auto& point : listOfPoints){
            array[counter] = point[0];
            array[counter+1] = point[1];
            array[counter+2] = point[2];
            counter += 3;
        }
        hsize_t chunk[3] = {size, 3};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, 3};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }

	static void writeArrayToFileId(const hid_t fileId, const std::string& dataset, Array3D<float>& array){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
		}
		const auto size = array.getSize();
		hsize_t chunk[3] = {size[0], size[1], size[2]};
		auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_deflate(dcpl, 9);
		H5Pset_chunk(dcpl, 3, chunk);

		hsize_t dims[3] = {size[0], size[1], size[2]};
		auto space = H5Screate_simple(3, dims, NULL);
		auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array.getData()[0]));
		H5Pclose(dcpl);
		H5Dclose(dset);
		H5Sclose(space);
	}

	static void writeStringToFileId(const hid_t fileId, const std::string& dataset, const std::string& data){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        hid_t dataspace_id = H5Screate (H5S_SCALAR);

        hid_t dtype = H5Tcopy (H5T_C_S1);
        int size = data.length();
        int status = H5Tset_size (dtype, size);
        char* metabuf = (char*) malloc(sizeof(char) * ( size + 1 ));
        strcpy (metabuf, data.c_str());
        metabuf[size] = '\0';

        hid_t dataset_id = H5Dcreate(fileId, dataset.c_str(), dtype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite (dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, metabuf);

        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        free(metabuf);
	}

	static void writeArrayToFileId(const hid_t fileId, const std::string& dataset, Array3D<unsigned short>& array){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
		}
		const auto size = array.getSize();
		hsize_t chunk[3] = {size[0], size[1], size[2]};
		auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_deflate(dcpl, 9);
		H5Pset_chunk(dcpl, 3, chunk);

		hsize_t dims[3] = {size[0], size[1], size[2]};
		auto space = H5Screate_simple(3, dims, NULL);
		auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U16LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
		H5Dwrite(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array.getData()[0]));
		H5Pclose(dcpl);
		H5Dclose(dset);
		H5Sclose(space);
	}

    static void writeArrayToFileId(const hid_t fileId, const std::string& dataset, Array3D<unsigned char>& array){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        const auto size = array.getSize();
        hsize_t chunk[3] = {size[0], size[1], size[2]};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 3, chunk);

        hsize_t dims[3] = {size[0], size[1], size[2]};
        auto space = H5Screate_simple(3, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U8LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array.getData()[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }


	static void writeArrayToFile(const std::string& filePath, Array3D<unsigned short>& voxelgrid, Array3D<unsigned short>& objClassGrid){
        hdf5_writer_mutex.lock();
		const auto fileId = openFile(filePath);
		Hdf5Writer::writeArrayToFileId(fileId, "voxelgrid", voxelgrid);
        Hdf5Writer::writeArrayToFileId(fileId, "objclassgrid", objClassGrid);
		H5Fclose(fileId);
        hdf5_writer_mutex.unlock();
	}

	static void writeArrayToFile(const std::string& filePath, Array3D<unsigned short>& voxelgrid){
        hdf5_writer_mutex.lock();
        const auto fileId = openFile(filePath);
        Hdf5Writer::writeArrayToFileId(fileId, "voxelgrid", voxelgrid);
        H5Fclose(fileId);
        hdf5_writer_mutex.unlock();
    }

    static void writeArrayToFile(const hid_t fileId, Array3D<std::list<DistPoint> >& space){
        for(unsigned int i = 0; i < space.getSize()[0]; ++i){
           for(unsigned int j = 0; j < space.getSize()[1]; ++j){
               for(unsigned int k = 0; k < space.getSize()[2]; ++k){
                   const auto& list = space(i,j,k);
                   if(!list.empty()) {
                       std::stringstream str;
                       str << "id_" << i << "_" << j << "_" << k;
                       Hdf5Writer::writeListToFileId(fileId, str.str(), list);
                   }
               }
           }
        }
	}

}

#endif //SDFGEN_HDF5WRITER_H
