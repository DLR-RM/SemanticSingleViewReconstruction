//
// Created by max on 20.10.21.
//

#include <random>
#include "Blocker.h"
#include <random>

Array3D<std::vector<std::vector<Point3D> >> Blocker::splitDataset(){
    // scale down the distances
    const double voxelSize = 2.0 / m_resolution;
    const double distScaleFac = 1.0 / voxelSize;
    const double boundarySize = (m_boundaryScaleFactorHalf - 1.0) * voxelSize;
    for(auto& dist: m_distances){
        dist *= distScaleFac;
    }

    // create all blocks by only creating list of indices
    Array3D<std::vector<int>> hardSelectionArray(m_resolution);
    for(int i = 0; i < m_points.size(); ++i){
        auto id = iPoint(((m_points[i] + 1) / voxelSize).floorThis());
        for(int k = 0; k < 3; ++k){
            if(id[k] >= m_resolution){
                id[k] = m_resolution - 1;
            }
        }
        hardSelectionArray(id[0], id[1], id[2]).emplace_back(i);
    }

    int usedPoints = 0;
    // split them with the boundary scale factor
    Array3D<std::vector<int>> softSelectionArray(m_resolution);
    for(int i = 0; i < m_resolution; ++i){
        for(int j = 0; j < m_resolution; ++j){
            for(int k = 0; k < m_resolution; ++k){
                const std::vector<int>& currentList = hardSelectionArray(i, j, k);
                if(!currentList.empty()){
                    softSelectionArray(i, j, k).reserve(int((double) currentList.size() * 1.5));
                    for(const auto id: currentList){
                        softSelectionArray(i, j, k).emplace_back(id);
                        ++usedPoints;
                    }
                    const dPoint coord = dPoint(i, j, k) / (m_resolution * 0.5) - 1;
                    const dPoint maxCoord = coord + voxelSize + boundarySize;
                    const dPoint minCoord = coord - boundarySize;
                    BoundingBox box(minCoord, maxCoord);
                    for(int l = -1; l < 2; ++l){
                        for(int m = -1; m < 2; ++m){
                            for(int n = -1; n < 2; ++n){
                                bool isMiddle = l == 0 && m == 0 && n == 0;
                                if(hardSelectionArray.isInCube(i+l, j+m, k+n) && !isMiddle){
                                    const std::vector<int>& neighBorList = hardSelectionArray(i+l, j+m, k+n);
                                    for(const auto neighborId: neighBorList){
                                        if(box.isInside(m_points[neighborId])){
                                            softSelectionArray(i, j, k).emplace_back(neighborId);
                                            ++usedPoints;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "Used points " << usedPoints << std::endl;

    // if not mode ALL filtering is done
    if(blockSelectionMode != "ALL"){
        const bool selectOnlyBoundary = blockSelectionMode == "BOUNDARY";
        int counter = 0;
        int removeCounter = 0;
        int removeBecauseNotFullEnough = 0;
        // check if block should be used or not
        for(int i = 0; i < m_resolution; ++i){
            for(int j = 0; j < m_resolution; ++j){
                for(int k = 0; k < m_resolution; ++k){
                    const std::vector<int>& currentList = softSelectionArray(i, j, k);
                    if(!currentList.empty()){
                        // check if enough points are in block
                        if(currentList.size() < m_minPointAmountPerBlock && selectOnlyBoundary){
                            // clear if not enough points are there
                            softSelectionArray(i, j, k).clear();
                            ++removeCounter;
                            ++removeBecauseNotFullEnough;
                            continue;
                        }else if(currentList.size() >= m_minPointAmountPerBlock && !selectOnlyBoundary){
                            // clear if not enough points are there
                            softSelectionArray(i, j, k).clear();
                            ++removeCounter;
                            ++removeBecauseNotFullEnough;
                            continue;
                        }

                        // find minimum and maximum distValue in current Block
                        auto minDist = DBL_MAX;
                        auto maxDist = -DBL_MAX;
                        auto absMinDist = DBL_MAX;
                        for(const auto l: currentList){
                            if(m_distances[l] < minDist){
                                minDist = m_distances[l];
                            }
                            if(m_distances[l] > maxDist){
                                maxDist = m_distances[l];
                            }
                            if(std::abs(m_distances[l]) < absMinDist){
                                absMinDist = std::abs(m_distances[l]);
                            }
                        }

                        // block selection mode is BOUNDARY, so only select blocks which have a boundary inside them
                        if(selectOnlyBoundary){
                            // check if signs are equal and if the absolute min distances is above the threshold
                            if((minDist < 0 == maxDist < 0) && absMinDist > m_truncationBoundarySelectionValue){
                                // do not use this block
                                softSelectionArray(i, j, k).clear();
                                ++removeCounter;
                            }else{
                                counter++;
                            }
                        }else{
                            // keep all blocks if non boundary is requested, except the blocks which are completely full
                            counter++;
                        }
                    }
                }
            }
        }
        std::cout << "Keep: " << counter << ", remove: " << removeCounter << ", removed because of not full enough: "
        << removeBecauseNotFullEnough<< ", before: " << counter + removeCounter << std::endl;
    }

    std::mt19937 rng(0);
    Array3D<std::vector<std::vector<Point3D> >> finalBlocks(m_resolution);
    // batch the final
    int batchCounterId = 0;
    for(int i = 0; i < m_resolution; ++i){
        for(int j = 0; j < m_resolution; ++j){
            for(int k = 0; k < m_resolution; ++k){
                const std::vector<int>& currentList = softSelectionArray(i, j, k);
                if(!currentList.empty()){
                    std::vector<std::vector<Point3D> >& currentBlocks = finalBlocks(i, j, k);
                    const dPoint coord = dPoint(i, j, k) / (m_resolution * 0.5) - 1;
                    const dPoint minCoord = coord - boundarySize;
                    for(int l = 0; l < currentList.size(); ++l){
                        int batchId = l / m_pointAmount;
                        if(batchId == currentBlocks.size()){
                            currentBlocks.emplace_back(std::vector<Point3D>());
                            currentBlocks[batchId].reserve(m_pointAmount);
                        }
                        const dPoint pos = ((m_points[currentList[l]] - minCoord) / (voxelSize * m_boundaryScaleFactor * 0.5) - 1.0) * m_boundaryScaleFactorHalf;
                        const Point3D newPoint(pos[0], pos[1], pos[2], m_classes[currentList[l]], m_distances[currentList[l]], batchCounterId);
                        currentBlocks[batchId].emplace_back(newPoint);
                    }
                    const auto currentBatchId = currentBlocks.size() - 1;
                    const int missingAmountOfPoints = m_pointAmount - (int) currentBlocks[currentBatchId].size();
                    std::uniform_int_distribution<int> gen(0, currentList.size() - 1); // uniform, unbiased
                    if(missingAmountOfPoints > 0){
                        for(int l = 0; l < missingAmountOfPoints; ++l){
                            const int randomId = gen(rng);
                            const int pointId = currentList[randomId];
                            const dPoint pos = ((m_points[pointId] - minCoord) / (voxelSize * m_boundaryScaleFactor * 0.5) - 1.0) * m_boundaryScaleFactorHalf;
                            const Point3D newPoint(pos[0], pos[1], pos[2], m_classes[currentList[randomId]], m_distances[currentList[randomId]], batchCounterId);
                            currentBlocks[currentBatchId].emplace_back(newPoint);
                        }
                    }
                    ++batchCounterId;
                }
            }
        }
    }
    return finalBlocks;
}
