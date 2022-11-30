/************************************************************

  This example shows how to read and write data to a dataset
  using gzip compression (also called zlib or deflate).  The
  program first checks if gzip compression is available,
  then if it is it writes integers to a dataset using gzip,
  then closes the file.  Next, it reopens the file, reads
  back the data, and outputs the type of compression and the
  maximum value in the dataset to the screen.

  This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include <iostream>
#include <vector>
#include "Array3D.h"
#include "Hdf5ReaderAndWriter.h"
#include "util/StopWatch.h"
#include "src/FilterClasses.h"
#include <tclap/CmdLine.h>
#include <thread>
#include <iterator>
#include <regex>
#include <cstdlib>

Array3D<double> average(const Array3D<double>& input, const int outputResolution){
	/**
	 * Averages an input array to a size of outputResolution x outputResolution x outputResolution.
	 */
	Array3D<double> res(outputResolution);
	const unsigned int blockSize = input.length() / outputResolution;
	const double amountOfBlocks = blockSize * blockSize * blockSize;
	for(unsigned int x = 0; x < outputResolution; ++x){
		for(unsigned int y = 0; y < outputResolution; ++y){
			for(unsigned int z = 0; z < outputResolution; ++z){
				double mean_value = 0;
				double counter = 1.0;
				for(unsigned int i = 0; i < blockSize; ++i){
					for(unsigned int j = 0; j < blockSize; ++j){
						for(unsigned int k = 0; k < blockSize; ++k){
							const double fac = 1.0 / counter;
							mean_value = fac * input(x * blockSize + i, y * blockSize + j,
													 z * blockSize + k) + (1.0 - fac) * mean_value;
							counter += 1;
						}
					}
				}
				res(x, y, z) = mean_value;
			}
		}
	}
	return res;
}

#define add_to_string(name) "\"" #name "\": " << name << ", "
#define add_to_string_char(name) "\"" #name "\": " << int(name) << ", "

static void
convertToLossValues(const Array3D<unsigned char>& lossValueTypes, Array3D<double>& result,
					const double wallValue, const double freeValue,
					const double wallValueNotVisible, const double notReachable,
					const double freeNotVisibleLower, const double freeNotVisibleUpper,
					std::string& stringConfigResult){
	for(int i = 0; i < lossValueTypes.length(); ++i){
		for(int j = 0; j < lossValueTypes.length(); ++j){
			for(int k = 0; k < lossValueTypes.length(); ++k){
				const unsigned char currentValue = lossValueTypes(i, j, k);
				if(currentValue == LossValueType::wallValue() ||
				   currentValue == LossValueType::trueWallValue()){
					result(i, j, k) = wallValue;
				}else if(currentValue == LossValueType::freeValue()){
					result(i, j, k) = freeValue;
				}else if(currentValue == LossValueType::notReachable()){
					result(i, j, k) = notReachable;
				}else if(currentValue == LossValueType::wallValueNotVisible() ||
						 currentValue == LossValueType::trueWallValueNotVisible()){
					result(i, j, k) = wallValueNotVisible;
				}else if(currentValue >= LossValueType::freeNotVisibleStart() &&
						 currentValue <= LossValueType::freeNotVisibleEnd()){
					const double factor =
							double(currentValue - LossValueType::freeNotVisibleStart()) /
							double(LossValueType::freeNotVisibleEnd() -
								   LossValueType::freeNotVisibleStart());
					result(i, j, k) = factor * (freeNotVisibleUpper - freeNotVisibleLower) +
									  freeNotVisibleLower;
				}else{
					std::cout << "The current value is unknown: " << (int) currentValue << std::endl;
					exit(1);
				}
			}
		}
	}
	std::stringstream str;
	str << "{";
	str << add_to_string(wallValue);
	str << add_to_string(freeValue);
	str << add_to_string(notReachable);
	str << add_to_string(wallValueNotVisible);
	for(int currentValue = LossValueType::freeNotVisibleStart();
		currentValue <= LossValueType::freeNotVisibleEnd(); ++currentValue){
		str << "\"freeNotVisible_" << currentValue << "\": " << currentValue << ", ";
	}
	for(int currentValue = 0; currentValue <= amountOfClasses(); ++currentValue){
		str << "\"classValue_" << currentValue << "\": "
			<< LossValueType::getClassValue(currentValue, amountOfClasses()) << ", ";
	}
	for(int currentValue = 0; currentValue <= amountOfClasses(); ++currentValue){
		str << "\"classValueNotVisible_" << currentValue << "\": "
			<< LossValueType::getClassValueNotVisible(currentValue, amountOfClasses()) << ", ";
	}
	str << "}";
	stringConfigResult = str.str();
}


void createLossMap(const std::string& file, const std::string& goalFile, unsigned int usedSize){
	/**
	 * Creates a loss map for the given .hdf5 file, the file must contain a "voxel_space" with a resolution of usedSize.
	 * The resulting loss map will be saved to the goalFile.
	 */
	StopWatch global;
	Array3D<double> voxel;
	std::vector<ClassPoint> points;
	const int usedClassSize = 64; // not 256 as then there are too many gaps
	Array3D<std::list<unsigned char> > classes(usedClassSize);
	Array3D<unsigned char> classesId(usedClassSize);
	classesId.fill(255);
	try{
		// at first the voxel grid will be read from the file
		Hdf5ReaderAndWriter::readVoxel(file, voxel, usedSize);
		const int amountOfPoints = 2000000;
		Hdf5ReaderAndWriter::readPoints(file, points, amountOfPoints);
		StopWatch globalClassPoints;
        bool is_replica_example = false;
        if(const char* env_p = std::getenv("IS_REPLICA")){
            is_replica_example = true;
        }

		for(const auto& point: points){
			auto p = dPoint(point);
			for(int i = 0; i < 3; ++i){
				p[i] = (p[i] * 0.5 + 0.5) * usedClassSize;
				if(int(p[i]) == usedClassSize){
					p[i] = usedClassSize - 1;
				}
			}
			const auto index = iPoint(p);
			unsigned int classId = 0;
            if(is_replica_example){
				classId = filterClassIdReplica(point.getClassId());
            }else{
				classId = filterClassId(point.getClassId());
            }
			classes(index[0], index[1], index[2]).emplace_back(classId);
		}
		std::vector<int> classesCounter;
		classesCounter.resize(amountOfClasses());
		for(int i = 0; i < classes.innerSize(); ++i){
			if(!classes.getInner(i).empty()){
				std::fill(classesCounter.begin(), classesCounter.end(), 0);
				for(const auto& ele: classes.getInner(i)){
					classesCounter[ele] += 1;
				}
				int highestId = -1;
				int highestCounter = 0;
				for(int j = 0; j < amountOfClasses(); ++j){
					if(classesCounter[j] > highestCounter){
						highestCounter = classesCounter[j];
						highestId = j;
					}
				}
				classesId.getInner(i) = highestId;
			}
		}
		std::cout << "Took: " << globalClassPoints.elapsed_time() << "s" << std::endl;
		// as the voxel grid is stored as 16 bit unsigned short it is converted back to a float
		// with a range from -1 to 1
		voxel.scaleToRange();

		Array3D<double> averaged_tsdf_32 = average(voxel, 32);
		Array3D<double> averaged_tsdf_16 = average(voxel, 16);
		Array3D<double> averaged_tsdf_8 = average(voxel, 8);

		Array3D<unsigned char> res(usedSize);
		// we then have to calculate the free space between the camera and the first obstacle
		// furthermore, we set the values of the wall, which are slightly before and after the first surface
		// the rest of the values are not changed
		voxel.projectWallValueIntoZ(res);
		// we now select a free position in the first plane (closest to the camera) to start our flood fill algorithm,
		// as it might be that there are obstacle intersecting with the near plane of the camera, so we have to search
		// for free space right in front of the camera
		auto pose = voxel.findFreeElementIn(2, 0);

		// from this pose out we perform a flood fill to find all voxels which are not visible from the camera, but are
		// not filled. We set the losses for the free space behind obstacles and we set the surfaces for all hidden
		// obstacles. Only obstacles, which can not be reached, because they are behind a hole free wall, are neglected.
		voxel.performFloodFill(pose, res);

		int counter = 0;
		const int factor = usedSize / usedClassSize;
		for(int i = 0; i < res.innerSize(); ++i){
			const auto pose3d = res.convertToPose(i);
			const unsigned char currentClassId = classesId(pose3d.first / factor, pose3d.second / factor, pose3d.third / factor);
			if(currentClassId < 255){
				if(res.getInner(i) == LossValueType::wallValue()){
					// if the res is a wall value and a class value
					res.getInner(i) = LossValueType::getClassValue(currentClassId, amountOfClasses());
					counter++;
				}else if(res.getInner(i) == LossValueType::wallValueNotVisible()){
					res.getInner(i) = LossValueType::getClassValueNotVisible(currentClassId,
																			 amountOfClasses());
					counter++;
				}else if(res.getInner(i) == LossValueType::trueWallValue()){
					res.getInner(i) = LossValueType::getClassValueTrueWall(currentClassId);
					counter++;
				}else if(res.getInner(i) == LossValueType::trueWallValueNotVisible()){
					res.getInner(i) = LossValueType::getClassValueNotVisibleTrueWall(currentClassId, amountOfClasses());
					counter++;
				}
			}
		}
		std::cout << "Changed values: " << counter << std::endl;

		//Array3D<double> resultValues(usedSize);
		//std::string usedLossValues;
		//convertToLossValues(res, resultValues, 100, 2, 20, 0.01, 0.25, 0.5, usedLossValues);

		//// we then reduce the size from usedSize to 32x32x32
		//Array3D<double> averaged_loss_32 = average(resultValues, 32);
		//Array3D<double> averaged_loss_16 = average(resultValues, 16);
		//Array3D<double> averaged_loss_8 = average(resultValues, 8);

		// and store it in the goalFile, which is also an .hdf5 container
		auto fileId = Hdf5ReaderAndWriter::openFile(goalFile, 't');
		//Hdf5ReaderAndWriter::writeArrayToFile(fileId, "lossmap_32", averaged_loss_32);
		//Hdf5ReaderAndWriter::writeArrayToFile(fileId, "lossmap_16", averaged_loss_16);
		//Hdf5ReaderAndWriter::writeArrayToFile(fileId, "lossmap_8", averaged_loss_8);
		Hdf5ReaderAndWriter::writeArrayToFile(fileId, "average_32", averaged_tsdf_32);
		Hdf5ReaderAndWriter::writeArrayToFile(fileId, "average_16", averaged_tsdf_16);
		Hdf5ReaderAndWriter::writeArrayToFile(fileId, "average_8", averaged_tsdf_8);
		Hdf5ReaderAndWriter::writeArrayToFile(fileId, "lossmap_valued", res);
		//Hdf5ReaderAndWriter::writeStringToFileId(fileId, "used_loss_values", usedLossValues);
		std::stringstream str;
		str << "{\"MAX_SEE_AROUND_EDGES_DIST\": " << MAX_SEE_AROUND_EDGES_DIST << ", ";
		str << add_to_string_char(LossValueType::wallValue());
		str << add_to_string_char(LossValueType::trueWallValue());
		str << add_to_string_char(LossValueType::freeValue());
		str << add_to_string_char(LossValueType::wallValueNotVisible());
		str << add_to_string_char(LossValueType::trueWallValueNotVisible());
		str << add_to_string_char(LossValueType::notReachable());
		for(int i = 0; i < MAX_SEE_AROUND_EDGES_DIST; ++i){
			str << "\"" << "LossValueType::freeNotVisible(" << i << ")\": "
				<< int(LossValueType::freeNotVisible(i)) << ", ";
		}
		for(int currentValue = 0; currentValue < amountOfClasses(); ++currentValue){
			str << "\"classValueTrueWall_" << currentValue << "\": "
				<< (int) LossValueType::getClassValueTrueWall(currentValue) << ", ";
		}
		for(int currentValue = 0; currentValue < amountOfClasses(); ++currentValue){
			str << "\"classValue_" << currentValue << "\": "
				<< (int) LossValueType::getClassValue(currentValue, amountOfClasses()) << ", ";
		}
		for(int currentValue = 0; currentValue < amountOfClasses(); ++currentValue){
			str << "\"classValueNotVisibleTrueWall_" << currentValue << "\": "
				<< (int) LossValueType::getClassValueNotVisibleTrueWall(currentValue, amountOfClasses()) << ", ";
		}
		for(int currentValue = 0; currentValue < amountOfClasses(); ++currentValue){
			str << "\"classValueNotVisible_" << currentValue << "\": "
				<< (int) LossValueType::getClassValueNotVisible(currentValue, amountOfClasses()) << ", ";
		}
		str << "}";
		std::string usedParams = str.str();
		Hdf5ReaderAndWriter::writeStringToFileId(fileId, "used_params", usedParams);
		H5Fclose(fileId);
		std::cout << "Done in " << global.elapsed_time() << ", " << goalFile << std::endl;
	}catch(UnusableException e){
		std::cout << "This file is broke: " << file << std::endl;
	}
}

class WordDelimitedByComma: public std::string {
};

std::istream& operator>>(std::istream& is, WordDelimitedByComma& output){
	std::getline(is, output, ',');
	return is;
}


int main(int argc, char** argv){
	TCLAP::CmdLine cmd("Loss Calculator for a given tsdf grid", ' ', "0.9");
	TCLAP::ValueArg<std::string> pathArg("p", "path",
										 "Paths to the files, separated by a comma, for each file a new thread is started",
										 true, "", "string");
	TCLAP::ValueArg<int> resolution("r", "resolution", "Used resolution 128, 256 or 512", true, 0,
									"int");

	cmd.add(pathArg);
	cmd.add(resolution);
	cmd.parse(argc, argv);

	if(pathArg.getValue().length() == 0){
		std::cout << "Set the path value!" << std::endl;
		exit(1);
	}

	const unsigned int usedSize = resolution.getValue();
	if(usedSize == 0){
		std::cout << "The resolution can not be zero" << std::endl;
		exit(1);
	}

	const std::string allPaths = pathArg.getValue();
	std::istringstream iss(allPaths);
	std::vector<std::string> files((std::istream_iterator<WordDelimitedByComma>(iss)),
								   std::istream_iterator<WordDelimitedByComma>());
	for(auto& filePath: files){
		filePath = std::regex_replace(filePath, std::regex("~"), std::string(getenv("HOME")));
	}
	std::vector<std::string> goalFiles;
	for(const auto& file: files){
		auto pos = file.rfind('.');
		goalFiles.emplace_back(file.substr(0, pos) + "_loss_avg.hdf5");
	}

	for(int i = 0; i < files.size(); ++i){
		createLossMap(files[i], goalFiles[i], usedSize);
	}
	return 0;
}
