#include <iostream>

#include "src/Hdf5ReaderAndWriter.h"
#include "src/StopWatch.h"
#include "src/Blocker.h"
#include <map>
#include <tclap/CmdLine.h>


int main(int argc, char** argv){
    TCLAP::CmdLine cmd("Blocks a given tsdf into blocks", ' ', "0.9");
	TCLAP::ValueArg<std::string> pathArg("p","path","Path to the file",true,"","string");
    TCLAP::ValueArg<std::string> pathGoalArg("g","goal_path","Path to goal file",true,"","string");
    std::string blockSelectionModeHelp = "Block selection mode: \"ALL\": all blocks are selected, \"BOUNDARY\": only the blocks "
                                         "on the boundary are selected, \"NON-BOUNDARY\": only the block not on the boundary are selected";
    TCLAP::ValueArg<std::string> blockSelectionModeTclap("b", "block_selection_mode", blockSelectionModeHelp, false, "BOUNDARY", "string");

	cmd.add(pathArg);
    cmd.add(pathGoalArg);
    cmd.add(blockSelectionModeTclap);
	cmd.parse(argc, argv);

    const std::string blockSelectionMode = blockSelectionModeTclap.getValue();
    if(!(blockSelectionMode == "ALL" || blockSelectionMode == "BOUNDARY" || blockSelectionMode == "NON-BOUNDARY")){
        std::cout << "The boundary selection mode can only be one of these three: \"ALL\": all blocks are selected, "
                     "\"BOUNDARY\": only the blocks on the boundary are selected, \"NON-BOUNDARY\": only the block not "
                     "on the boundary are selected. And not: \"" << blockSelectionMode << "\"" << std::endl;
        exit(1);
    }

    const std::string filePath = pathArg.getValue();
    hid_t fileId = Hdf5ReaderAndWriter::openFile(filePath, 'r');
    std::vector<dPoint> points;
    Hdf5ReaderAndWriter::readVector(fileId, "points", points, 2000000);
    std::vector<double> distances;
    Hdf5ReaderAndWriter::readVector(fileId, "distances", distances, 2000000);
    std::vector<unsigned char> classes;
    Hdf5ReaderAndWriter::readVector(fileId, "classes", classes, 2000000);
    H5Fclose(fileId);



    auto sw2 = StopWatch();
    Blocker blocker(std::move(points), std::move(distances), std::move(classes), blockSelectionMode);
    Array3D<std::vector<std::vector<Point3D> >> finalBlocks = blocker.splitDataset();

    std::vector<std::vector<Point3D> > listOfPoints;
    std::map<int, iPoint> batchCounterMap;
    for(int i = 0; i < finalBlocks.length(); ++i){
        for(int j = 0; j < finalBlocks.length(); ++j){
            for(int k = 0; k < finalBlocks.length(); ++k){
                for(const auto& batchList: finalBlocks(i, j, k)){
                    listOfPoints.emplace_back(std::vector<Point3D>());
                    const int currentIndex = listOfPoints.size() - 1;
                    for(const auto& point: batchList){
                        const auto batchIt = batchCounterMap.find(point.getBatchCounter());
                        if(batchIt == batchCounterMap.end()){
                            batchCounterMap[point.getBatchCounter()] = iPoint (i, j, k);
                        }
                        listOfPoints[currentIndex].emplace_back(point);
                    }
                }
            }
        }
    }
    std::vector<std::array<int, 4> > batchCounterMapArray;
    batchCounterMapArray.reserve(batchCounterMap.size());
    for(const auto ele: batchCounterMap){
        batchCounterMapArray.emplace_back(std::array<int, 4>{ele.first, ele.second[0], ele.second[1], ele.second[2]});
    }
    std::cout << "Splitting done in: " << sw2.elapsed_time() << std::endl;

    std::cout << "Amount of blocks: " << listOfPoints.size() << std::endl;

    auto sw3 = StopWatch();
    hid_t fileIdWrite = Hdf5ReaderAndWriter::openFile(pathGoalArg.getValue(), 't');
    Hdf5ReaderAndWriter::writeVectorPointDistToFileId(fileIdWrite, "points_dist", listOfPoints);
    Hdf5ReaderAndWriter::writeVectorClassBatchToFileId(fileIdWrite, "class_batch_counter", listOfPoints);
    Hdf5ReaderAndWriter::writeVectorToFileId(fileIdWrite, "batch_counter_map", batchCounterMapArray);
    H5Fclose(fileIdWrite);
    std::cout << "Writing done in: " << sw2.elapsed_time() << std::endl;




    return 0;
}
