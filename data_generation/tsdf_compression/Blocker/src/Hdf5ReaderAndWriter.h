//
// Created by Maximilian Denninger on 2019-06-17.
//

#ifndef LOSSCALCULATOR_HDF5READERANDWRITER_H
#define LOSSCALCULATOR_HDF5READERANDWRITER_H


#include <cstring>
#include <string>
#include <vector>
#include "hdf5.h"
#include "Point.h"
#include "Array3D.h"

class UnusableException : public std::exception {
	virtual const char* what() const throw() {
    	return "This file can not be used at all!";
  	}
};

class Hdf5ReaderAndWriter {
public:

	static hid_t openFile(const std::string& filePath, const char mode){
	    hid_t fileId;
		if(mode == 'r'){
			fileId = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		}else if(mode == 't'){
            fileId = H5Fcreate(filePath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		}else{
            fileId = H5Fopen(filePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		}
		return fileId;
	}

	static hid_t openReadDataSet(const hid_t fileId, const std::string& dataset){
		if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
			return H5Dopen(fileId, dataset.c_str(), H5P_DEFAULT);
		}
        throw UnusableException();
    }

    static void readVector(const hid_t fileId, const std::string& dataset, std::vector<unsigned char>& array, unsigned int size){
        hid_t dset = openReadDataSet(fileId, dataset);

        const auto maxNr = size;
        auto* inner_array = new unsigned char[maxNr];
		H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, (void*) inner_array);
		H5Dclose(dset);
        array.resize(size);
        for(unsigned int i = 0; i < size; ++i){
            array[i] = inner_array[i];
        }
		delete[](inner_array);
	}

    static void readVector(const hid_t fileId, const std::string& dataset, std::vector<double>& array, unsigned int size){
        hid_t dset = openReadDataSet(fileId, dataset);

        const auto maxNr = size;
        auto* inner_array = new float[maxNr];
		H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, (void*) inner_array);
		H5Dclose(dset);
        array.resize(size);
        for(unsigned int i = 0; i < size; ++i){
            array[i] = (double) inner_array[i];
        }
		delete[](inner_array);
	}

    static void readVector(const hid_t fileId, const std::string& dataset, std::vector<dPoint>& array, unsigned int size){
        hid_t dset = openReadDataSet(fileId, dataset);

        const auto maxNr = size * 3;
        auto* inner_array = new float[maxNr];
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, (void*) inner_array);
        H5Dclose(dset);
        array.resize(size);
        int counter = 0;
        for(unsigned int i = 0; i < size; ++i){
            for(std::size_t j = 0; j < 3; ++j){
                array[i][j] = (double) inner_array[counter];
                ++counter;
            }
        }
        delete[](inner_array);
    }

     static void writeVectorPointDistToFileId(const hid_t fileId, const std::string dataset, const std::vector<std::vector<Point3D> >& listOfPoints){
         if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT) > 0){
             H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
         }
         std::vector<float> array;
         const auto size = listOfPoints.size();
         const int amountOfEle = 4;
         const hsize_t amountOfPoints = listOfPoints[0].size();
         array.resize(size * amountOfPoints * amountOfEle);
         int counter = 0;
         for(const auto &batchBlock: listOfPoints){
             for(const auto &point: batchBlock){
                 array[counter] = point[0];
                 array[counter + 1] = point[1];
                 array[counter + 2] = point[2];
                 array[counter + 3] = point.getDistance();
                 counter += amountOfEle;
             }
         }
         hsize_t chunk[3] = {size, amountOfPoints, amountOfEle};
         auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
         //H5Pset_deflate(dcpl, 9);
         H5Pset_chunk(dcpl, 3, chunk);

         hsize_t dims[3] = {size, amountOfPoints, amountOfEle};
         auto space = H5Screate_simple(3, dims, NULL);
         auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_IEEE_F32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
         H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
         H5Pclose(dcpl);
         H5Dclose(dset);
         H5Sclose(space);
     }

    static void writeVectorClassBatchToFileId(const hid_t fileId, const std::string dataset, const std::vector<std::vector<Point3D> >& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        std::vector<int> array;
        const auto size = listOfPoints.size();
        const int amountOfEle = 2;
        const hsize_t amountOfPoints = listOfPoints[0].size();
        array.resize(size * amountOfPoints * amountOfEle);
        int counter = 0;
        for(const auto &batchBlock: listOfPoints){
            for(const auto &point: batchBlock){
                array[counter] = point.getClass();
                array[counter + 1] = point.getBatchCounter();
                counter += amountOfEle;
            }
        }
        hsize_t chunk[3] = {size, amountOfPoints, amountOfEle};
        auto dcpl = H5Pcreate(H5P_DATASET_CREATE);
        //H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 3, chunk);

        hsize_t dims[3] = {size, amountOfPoints, amountOfEle};
        auto space = H5Screate_simple(3, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
        H5Pclose(dcpl);
        H5Dclose(dset);
        H5Sclose(space);
    }

    template<std::size_t amount>
    static void writeVectorToFileId(const hid_t fileId, const std::string dataset, const std::vector<std::array<int, amount> >& listOfPoints){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }

        std::vector<int> array;
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
        //H5Pset_deflate(dcpl, 9);
        H5Pset_chunk(dcpl, 2, chunk);

        hsize_t dims[2] = {size, amount};
        auto space = H5Screate_simple(2, dims, NULL);
        auto dset = H5Dcreate(fileId, dataset.c_str(), H5T_STD_U32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(array[0]));
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


    static void writeStringToFileId(const hid_t fileId, const std::string& dataset, const std::string& data){
        if(H5Lexists(fileId, dataset.c_str(), H5P_DEFAULT ) > 0){
            H5Ldelete(fileId, dataset.c_str(), H5P_DEFAULT);
        }
        hid_t dataspace_id = H5Screate (H5S_SCALAR);

        hid_t dtype = H5Tcopy (H5T_C_S1);
        int size = data.length();
        H5Tset_size (dtype, size);
        char* metabuf = (char*) malloc(sizeof(char) * ( size + 1 ));
        strcpy (metabuf, data.c_str());
        metabuf[size] = '\0';

        hid_t dataset_id = H5Dcreate(fileId, dataset.c_str(), dtype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite (dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, metabuf);

        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        free(metabuf);
	}
};


#endif //LOSSCALCULATOR_HDF5READERANDWRITER_H
