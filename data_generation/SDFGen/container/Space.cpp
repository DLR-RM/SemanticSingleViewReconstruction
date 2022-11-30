//
// Created by Maximilian Denninger on 15.08.18.
//

#include <cfloat>
#include "Space.h"
#include "../util/StopWatch.h"
#include "../geom/math/AvgNumber.h"
#include "Octree.h"
#include <chrono>

void Space::internCalcDistApproximately(const Array3D<std::vector<Polygon*>*>& initialContainedPolygons, Array3D<Polygon*>& closestPolygons, const Polygons &polygons, unsigned int size, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy) {
    std::queue<uiPoint> openList;

    // Mark voxels which contain polygons
    for (unsigned int x = 0; x < size; x++) {
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int z = 0; z < size; z++) {
                if (initialContainedPolygons(x, y, z) != nullptr) {
                    openList.emplace(x, y, z);
                }
            }
        }
    }

    // Start flood fill from these voxels and mark every voxel with the min dist to the next voxel which contains a polygon
    std::vector<Polygon*> polygonChoices;
    while (!openList.empty()) {
        const uiPoint& current = openList.front();

        for (int x = (current[0] == 0 ? 0 : -1); x <= (current[0] == size - 1 ? 0 : 1); x++) {
            for (int y = (current[1] == 0 ? 0 : -1); y <= (current[1] == size - 1 ? 0 : 1); y++) {
                for (int z = (current[2] == 0 ? 0 : -1); z <= (current[2] == size - 1 ? 0 : 1); z++) {
                    if (closestPolygons(current[0] + x, current[1] + y, current[2] + z) == nullptr) {
                        openList.emplace(current[0] + x, current[1] + y, current[2] + z);
                        fillVoxel(initialContainedPolygons, closestPolygons, openList.back(), polygonChoices, size, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);
                    }
                }
            }
        }

        openList.pop();
    }
}

void Space::fillVoxel(const Array3D<std::vector<Polygon*>*>& initialContainedPolygons, Array3D<Polygon*>& closestPolygons, const uiPoint& voxel, std::vector<Polygon*>& polygonChoices, unsigned int size, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy) {
    polygonChoices.clear();

    // Collect the closest polygons of the voxels in the neighbourhood
    bool useThreshold = true;
    double currentThreshold = 0;
    int currentObjClass = 0;

    for (int x = std::max(0, (int)voxel[0] - approximationAccuracy); x <= std::min(size - 1, voxel[0] + approximationAccuracy); x++) {
        for (int y = std::max(0, (int)voxel[1] - approximationAccuracy); y <= std::min(size - 1, voxel[1] + approximationAccuracy); y++) {
            for (int z = std::max(0, (int)voxel[2] - approximationAccuracy); z <= std::min(size - 1, voxel[2] + approximationAccuracy); z++) {

                if (initialContainedPolygons(x, y, z) != nullptr || closestPolygons(x, y, z) != nullptr) {
                    if (initialContainedPolygons(x, y, z) != nullptr) {
                        for (Polygon *poly : *initialContainedPolygons(x, y, z)) {
                            polygonChoices.push_back(poly);
                        }
                    } else if (closestPolygons(x, y, z) != nullptr) {
                        polygonChoices.push_back(closestPolygons(x, y, z));
                    }

                    if (truncationThreshold == 0 || fabs(m_data(x, y, z)) < truncationThreshold)
                        useThreshold = false;
                    else {
                        currentThreshold = m_data(x, y, z);
                        currentObjClass = m_class(x, y, z);
                    }

                }
            }
        }
    }

    if (polygonChoices.size() == 0) {
        printMsg("No neighbor voxels contained a pointer to their closest polygons. This should not happen.");
        exit(0);
    }

    // If all neighbour voxels contain only the trunc threshold, then we can directly also use the threshold for this voxels
    if (useThreshold) {
        closestPolygons(voxel[0], voxel[1], voxel[2]) = polygonChoices.front();
        m_data(voxel[0], voxel[1], voxel[2]) = currentThreshold;
        m_class(voxel[0], voxel[1], voxel[2]) = currentObjClass;
        return;
    }

    // Go through all collected polygons and find the one with minimum distance
    Polygon *closestPosPoly = nullptr;
    Polygon *closestPoly = nullptr;
    double minDist = DBL_MAX;
    double minPosDist = DBL_MAX;
    dPoint point = getCenterOf(voxel[0], voxel[1], voxel[2]);
    for (Polygon *poly : polygonChoices) {
        double dist = poly->calcDistanceConst(point);
        if (fabs(dist) < fabs(minDist)) {
            minDist = dist;
            closestPoly = poly;
        }
        if (dist > 0 && dist < minPosDist) {
            minPosDist = dist;
            closestPosPoly = poly;
        }
    }
    if (minPosDist != DBL_MAX && fabs(minPosDist - fabs(minDist)) < maxDistanceToMinPosDist) {
        m_data(voxel[0], voxel[1], voxel[2]) = fabs(minDist);
        m_class(voxel[0], voxel[1], voxel[2]) = closestPosPoly->getObjectClass();
        closestPolygons(voxel[0], voxel[1], voxel[2]) = closestPosPoly;
    } else {
        m_data(voxel[0], voxel[1], voxel[2]) = minDist;
        m_class(voxel[0], voxel[1], voxel[2]) = closestPoly->getObjectClass();
        closestPolygons(voxel[0], voxel[1], voxel[2]) = closestPoly;
    }

    if (truncationThreshold != 0) {
        m_data(voxel[0], voxel[1], voxel[2]) = std::min((float) truncationThreshold,
                                                        std::max(-(float) truncationThreshold,
                                                                 m_data(voxel[0], voxel[1], voxel[2])));
        if (m_data(voxel[0], voxel[1], voxel[2]) == truncationThreshold) {
            m_class(voxel[0], voxel[1], voxel[2]) = 0;
        }
    }
}

void Space::internCalcDistExactly(const Octree &octree, unsigned int start, unsigned int end, int totalNumberOfLeafs, double maxDistanceToMinPosDist, double truncationThreshold) {
    const auto size = m_data.getSize();
    std::vector<QueueElement> openList;
    std::vector<const Octree *> neighbours;
    std::vector<bool> visitedNodes(totalNumberOfLeafs);
    openList.reserve(totalNumberOfLeafs);
    const Polygon *closestPoly;

    for (unsigned int i = start; i < end; ++i) {
        printVar(i);
        for (unsigned int j = 0; j < size[1]; ++j) {
            for (unsigned int k = 0; k < size[2]; ++k) {
                closestPoly = nullptr;
                m_data(i, j, k) = interCalcDistExactlyForPoint(getCenterOf(i, j, k), octree, openList, neighbours, visitedNodes, closestPoly, maxDistanceToMinPosDist);

                if (truncationThreshold != 0) {
                    m_data(i, j, k) = std::min((float) truncationThreshold,
                                               std::max(-(float) truncationThreshold, m_data(i, j, k)));
                }
            }
        }
    }
}

double Space::interCalcDistExactlyForPoint(const dPoint &point, const Octree &octree, std::vector<QueueElement> &openList, std::vector<const Octree *> &neighbours, std::vector<bool> &visitedNodes, const Polygon *&closestPoly, double maxDistanceToMinPosDist) {

    openList.clear();
    openList.emplace_back(0, &octree.findNodeContainingPoint(point));
    visitedNodes[openList[0].node->getId()] = true;

    double minDist = DBL_MAX;
    double minPosDist = DBL_MAX;
    double minimumMaxDistToFilledNode = DBL_MAX;

    const double truncationThreshold = DBL_MAX;

    for (int currentIndex = 0; currentIndex < openList.size(); currentIndex++) {
        const auto &current = openList[currentIndex];

        if (current.minDistanceToBlock > fabs(minDist) + maxDistanceToMinPosDist ||
            current.minDistanceToBlock > fabs(minimumMaxDistToFilledNode) + maxDistanceToMinPosDist ||
            fabs(current.minDistanceToBlock) > truncationThreshold) {
            continue;
        }

        if (current.node->getState() == OCTREE_FILLED) {
            for (auto poly : current.node->getIntersectingPolygons()) {
                double dist = poly->calcDistanceConst(point);

                if (fabs(dist) < fabs(minDist)) {
                    minDist = dist;
                    closestPoly = poly;
                }
                if (dist > 0 && dist < minPosDist) {
                    minPosDist = dist;
                    closestPoly = poly;
                }
            }
        }

        neighbours.clear();
        current.node->findLeafNeighbours(neighbours);

        for (const auto &neighbour : neighbours) {
            if (!visitedNodes[neighbour->getId()]) {
                double dist = neighbour->calcMinDistanceToPoint(point);
                if (dist <= fabs(minDist) + maxDistanceToMinPosDist) {
                    if (neighbour->getState() == OCTREE_FILLED) {
                        minimumMaxDistToFilledNode = std::min(minimumMaxDistToFilledNode, neighbour->calcMaxDistanceToPoint(point));
                    }
                }
                visitedNodes[neighbour->getId()] = true;
                openList.emplace_back(dist, neighbour);
            }
        }

    }

    for (auto &element : openList) {
        visitedNodes[element.node->getId()] = false;
    }

    if (minPosDist != DBL_MAX && fabs(minPosDist - fabs(minDist)) < maxDistanceToMinPosDist) {
        return fabs(minDist);
    } else if (minDist == DBL_MAX) {
        return truncationThreshold;
    } else {
        return minDist;
    }
}


Octree Space::createOctree(const Polygons &polys, double minimumOctreeVoxelSize) {
    MinMaxValue<double> minMax(-1, 1);
    for (const auto &poly : polys) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                minMax.add(poly.getPoints()[i][j]);;
            }
        }
    }

    Octree octree({0, 0, 0}, {2, 2, 2});
    octree.changeState(OCTREE_MIXED);
    for (const auto &poly : polys) {
        octree.addIntersectingPolygon(poly);
    }

    int maximumOctreeDepth = (int) round(log2(2 / minimumOctreeVoxelSize));
    buildOctree(octree, maximumOctreeDepth);
    printMsg("Finished octree! (depth: " + std::to_string(maximumOctreeDepth) + ")");
    return octree;
}

void Space::calcDistsApproximately(Polygons &polys, double maxDistanceToMinPosDist, double truncationThreshold, int approximationAccuracy) {
    unsigned int size = m_data.getSize()[0];
    Array3D<std::vector<Polygon*>*> initialContainedPolygons(m_data.getSize());

    {
        std::vector<const Octree *> leafs;
        Octree octree = createOctree(polys, 2.0 / size);

        octree.collectLeafChilds(leafs);

        for (const Octree *leaf : leafs) {
            if (leaf->getState() == OCTREE_FILLED) {
                for (Polygon *poly : leaf->getIntersectingPolygons()) {
                    const uiPoint &voxel({(unsigned int) ((leaf->getOrigin()[0] - -1) / (2.0 / size)), (unsigned int) ((leaf->getOrigin()[1] - -1) / (2.0 / size)), (unsigned int) ((leaf->getOrigin()[2] - -1) / (2.0 / size))});

                    if (initialContainedPolygons(voxel[0], voxel[1], voxel[2]) == nullptr)
                        initialContainedPolygons(voxel[0], voxel[1], voxel[2]) = new std::vector<Polygon *>();
                    initialContainedPolygons(voxel[0], voxel[1], voxel[2])->push_back(poly);
                }
            }
        }
    } // Octree should here be removed from memory

    StopWatch sw;

    for (auto &poly : polys) {
        poly.calcNormal();
    }

    Array3D<Polygon*> closestPolygons(m_data.getSize());

    internCalcDistApproximately(initialContainedPolygons, closestPolygons, polys, size, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);
    printVar(sw.elapsed_time());

    for (unsigned int x = 0; x < size; x++) {
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int z = 0; z < size; z++) {
                if (initialContainedPolygons(x, y, z) != nullptr) {
                    delete initialContainedPolygons(x, y, z);
                }
            }
        }
    }

}


void Space::calcDistsExactly(Polygons &polys, double minimumOctreeVoxelSize, double maxDistanceToMinPosDist, unsigned int amountOfThreads, double truncationThreshold) {
    Octree octree = createOctree(polys, minimumOctreeVoxelSize);

    printVar(polys.size());
    StopWatch sw;

    const auto size = m_data.getSize();
    int totalNumberOfLeafs = octree.resetIds();
    if (amountOfThreads == 0)
        amountOfThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (auto &poly : polys) {
        poly.calcNormal();
    }

    if (polys.size() > 0) {
        amountOfThreads = std::min(amountOfThreads, (unsigned int) polys.size());
        for (unsigned int i = 0; i < amountOfThreads; ++i) {
            unsigned int start = (unsigned int) (i * size[0] / (float) amountOfThreads);
            unsigned int end = (unsigned int) ((i + 1) * size[0] / (float) amountOfThreads);
            if (i + 1 == amountOfThreads) {
                end = size[0];
            }
            threads.emplace_back(
                    std::thread(&Space::internCalcDistExactly, this, std::cref(octree), start, end, totalNumberOfLeafs, maxDistanceToMinPosDist, truncationThreshold)
            );
        }
    } else {
        m_data.fill(DBL_MAX);
    }
    for (auto &thread : threads) {
        thread.join();
    }
    printVar(sw.elapsed_time());
}

dPoint Space::getCenterOf(unsigned int i, unsigned int j, unsigned int k) {
    dPoint res;
    const auto size = m_data.getSize();
    res[0] = (i + 0.5) / (double) size[0] * m_size[0] + m_origin[0];
    res[1] = (j + 0.5) / (double) size[1] * m_size[1] + m_origin[1];
    res[2] = (k + 0.5) / (double) size[2] * m_size[2] + m_origin[2];
    return res;
}

bool Space::isPointInSpace(const iPoint& point){
    const auto size = m_data.getSize();
    return point[0] >= 0 && point[1] >= 0 && point[2] >= 0 && point[0] < size[0] && point[1] < size[1] && point[2] < size[2];
}

void Space::fillNegativeVoxelsAndAddBorder(Array3D<VoxelLocation> &voxelLocations, int borderWidth) {
    const auto &size = m_data.getSize();
    uiPoint neighbour;
    for (unsigned int i = 0; i < size[0]; ++i) {
        for (unsigned int j = 0; j < size[1]; ++j) {
            for (unsigned int k = 0; k < size[2]; ++k) {
                if (m_data(i, j, k) <= 0) {
                    voxelLocations(i, j, k) = VOXEL_LOCATION_INSIDE;

                    for (neighbour[0] = std::max((int) i - borderWidth, 0); neighbour[0] <= std::min(i + borderWidth, size[0] - 1); neighbour[0]++) {
                        for (neighbour[1] = std::max((int) j - borderWidth, 0); neighbour[1] <= std::min(j + borderWidth, size[1] - 1); neighbour[1]++) {
                            for (neighbour[2] = std::max((int) k - borderWidth, 0); neighbour[2] <= std::min(k + borderWidth, size[2] - 1); neighbour[2]++) {
                                if (m_data(neighbour[0], neighbour[1], neighbour[2]) > 0) {
                                    voxelLocations(neighbour[0], neighbour[1], neighbour[2]) = VOXEL_LOCATION_BORDER;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Finds all visible (positive tsdf value) voxels that can be reached from the start point without having to cross a border voxel or an inside voxel
 */
void Space::floodFillVisibleVoxels(Array3D<VoxelLocation> &voxelLocations, int borderWidth) {
    const auto &size = m_data.getSize();
    std::vector<uiPoint> openList;
    // Add the start point
    openList.emplace_back(size[0] / 2, size[1] / 2, 0);
    voxelLocations(size[0] / 2, size[1] / 2, 0) = VOXEL_LOCATION_OUTSIDE;

    int index = 0;
    while (index < openList.size()) {
        uiPoint current = openList[index];

        // Add all the neighbors that are not previously visited and are neither INSIDE nor BOUNDARY voxel to the queue
        for (int x = (current[0] == 0 ? 0 : -1); x <= (current[0] == size[0] - 1 ? 0 : 1); x++) {
            for (int y = (current[1] == 0 ? 0 : -1); y <= (current[1] == size[1] - 1 ? 0 : 1); y++) {
                for (int z = (current[2] == 0 ? 0 : -1); z <= (current[2] == size[2] - 1 ? 0 : 1); z++) {
                    uiPoint neighbour(current[0] + x, current[1] + y, current[2] + z);

                    if (m_data(current[0], current[1], current[2]) > 2e-2) {
                        m_class(current[0], current[1], current[2]) = 0;
                    }

                    // Check the TSDF value condition and that the neighbours dont have an INSIDE or BOUNDARY state
                    if (m_data(current[0], current[1], current[2]) > 5e-2 || m_data(current[0], current[1], current[2]) >= m_data(neighbour[0], neighbour[1], neighbour[2])) {
                        if (voxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_AMBIGUOUS) {
                            voxelLocations(neighbour[0], neighbour[1], neighbour[2]) = VOXEL_LOCATION_OUTSIDE;
                            openList.emplace_back(neighbour);
                        }
                    }
                }
            }
        }

        index++;
    }
}

/**
 * Removes the border voxels which were previously added
 * But while removing we check if the border voxel has a neighbour(upto borderwidth) which is marked as OUTSIDE(visible voxel) previously
 * This is to make sure that we mark voxels which are marked as border previously but are inside an object and hence not visible from outside to AMBIGUOUS
 */
void Space::removeOutsideBorder(Array3D<VoxelLocation> &voxelLocations, int borderWidth) {
    const auto &size = m_data.getSize();
    Array3D<VoxelLocation> prevVoxelLocations = voxelLocations;
    for (unsigned int i = 0; i < size[0]; ++i) {
        for (unsigned int j = 0; j < size[1]; ++j) {
            for (unsigned int k = 0; k < size[2]; ++k) {
                if (prevVoxelLocations(i, j, k) == VOXEL_LOCATION_BORDER) {

                    std::vector<std::pair<int, uiPoint>> openList;
                    openList.emplace_back(0, uiPoint(i, j, k));

                    int index = 0;
                    bool found = false;

                    // Finding the neighbour(in a radius of borderwidth) which is marked as visible during the flood fill
                    while (index < openList.size()) {
                        uiPoint current = openList[index].second;
                        for (int x = (current[0] == 0 ? 0 : -1); x <= (current[0] == size[0] - 1 ? 0 : 1); x++) {
                            for (int y = (current[1] == 0 ? 0 : -1); y <= (current[1] == size[1] - 1 ? 0 : 1); y++) {
                                for (int z = (current[2] == 0 ? 0 : -1); z <= (current[2] == size[2] - 1 ? 0 : 1); z++) {
                                    uiPoint neighbour(current[0] + x, current[1] + y, current[2] + z);

                                    if (fabs(m_data(current[0], current[1], current[2])) < m_data(neighbour[0], neighbour[1], neighbour[2])) {
                                        if (prevVoxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_OUTSIDE) {
                                            found = true;
                                            goto next;
                                        } else if (prevVoxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_BORDER && openList[index].first + 1 < borderWidth) {
                                            openList.emplace_back(std::make_pair(openList[index].first + 1, neighbour));
                                        }
                                    }
                                }
                            }
                        }

                        index++;
                    }

                    next:;
                    voxelLocations(i, j, k) = (found ? VOXEL_LOCATION_OUTSIDE : VOXEL_LOCATION_AMBIGUOUS);
                    // if (found) m_class(i, j, k) = 0;
                }
            }
        }
    }
}

void Space::recalculateInnerVoxel(const uiPoint &pos, const uiPoint &size, Array3D<VoxelLocation> &voxelLocations, double truncationThreshold) {
    double minDist = -DBL_MAX;
    dPoint posPoint = getCenterOf(pos[0], pos[1], pos[2]);
    int closestObjClass = 0;
    int lookaround = 2;
    uiPoint neighbour;
    for (neighbour[0] = std::max((int) pos[0] - lookaround, 0); neighbour[0] <= std::min(pos[0] + lookaround, size[0] - 1); neighbour[0]++) {
        for (neighbour[1] = std::max((int) pos[1] - lookaround, 0); neighbour[1] <= std::min(pos[1] + lookaround, size[1] - 1); neighbour[1]++) {
            for (neighbour[2] = std::max((int) pos[2] - lookaround, 0); neighbour[2] <= std::min(pos[2] + lookaround, size[2] - 1); neighbour[2]++) {
                dPoint neighbourPoint = getCenterOf(neighbour[0], neighbour[1], neighbour[2]);

                double dist = -DBL_MAX;
                if (voxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_OUTSIDE || voxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_INSIDE_RECALCULATED) {
                    if (m_data(neighbour[0], neighbour[1], neighbour[2]) > 0) {
                        if (m_data(neighbour[0], neighbour[1], neighbour[2]) == truncationThreshold)
                            dist = truncationThreshold;
                        else
                            dist = -((posPoint - neighbourPoint).length() - m_data(neighbour[0], neighbour[1], neighbour[2]));
                    } else {
                        if (m_data(neighbour[0], neighbour[1], neighbour[2]) == -truncationThreshold)
                            dist = -truncationThreshold;
                        else
                            dist = -(posPoint - neighbourPoint).length() + m_data(neighbour[0], neighbour[1], neighbour[2]);
                    }
                }

                minDist = std::max(minDist, dist);
                if(minDist == dist) {
                    closestObjClass = m_class(neighbour[0], neighbour[1], neighbour[2]);
                }

            }
        }
    }

    m_data(pos[0], pos[1], pos[2]) = minDist;
    if (closestObjClass != 0 && m_class(pos[0], pos[1], pos[2]) == 0) m_class(pos[0], pos[1], pos[2]) = closestObjClass;
    voxelLocations(pos[0], pos[1], pos[2]) = VOXEL_LOCATION_INSIDE_RECALCULATED;
}


int Space::recalculateInnerVoxels(Array3D<VoxelLocation> &voxelLocations, double truncationThreshold) {
    const auto &size = m_data.getSize();

    std::vector<uiPoint> openList;
    for (unsigned int i = 0; i < size[0]; ++i) {
        for (unsigned int j = 0; j < size[1]; ++j) {
            for (unsigned int k = 0; k < size[2]; ++k) {
                if (voxelLocations(i, j, k) == VOXEL_LOCATION_OUTSIDE) {
                    openList.emplace_back(i, j, k);
                }
            }
        }
    }

    int index = 0;
    int numberOfRecalculatedVoxels = 0;
    while (index < openList.size()) {
        uiPoint current = openList[index];
        for (int x = (current[0] == 0 ? 0 : -1); x <= (current[0] == size[0] - 1 ? 0 : 1); x++) {
            for (int y = (current[1] == 0 ? 0 : -1); y <= (current[1] == size[1] - 1 ? 0 : 1); y++) {
                for (int z = (current[2] == 0 ? 0 : -1); z <= (current[2] == size[2] - 1 ? 0 : 1); z++) {
                    uiPoint neighbour(current[0] + x, current[1] + y, current[2] + z);

                    if (voxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_AMBIGUOUS || voxelLocations(neighbour[0], neighbour[1], neighbour[2]) == VOXEL_LOCATION_INSIDE) {
                        recalculateInnerVoxel(neighbour, size, voxelLocations, truncationThreshold);
                        openList.emplace_back(neighbour);
                        numberOfRecalculatedVoxels++;
                    }
                }
            }
        }

        index++;
    }

    //for (unsigned int i = 0; i < size[0]; ++i) {
    //    for (unsigned int j = 0; j < size[1]; ++j) {
    //        for (unsigned int k = 0; k < size[2]; ++k) {
    //            if (m_data(i, j, k) <= 0 && m_class(i, j, k) == 0) {
    //                printMsg("Found an inside voxel without a classs");
    //            }
    //        }
    //    }
    //}

    return numberOfRecalculatedVoxels;
}

void Space::correctInnerVoxel(int borderWidth, double truncationThreshold) {
    const auto &size = m_data.getSize();
    Array3D<VoxelLocation> voxelLocations(size);
    voxelLocations.fill(VOXEL_LOCATION_AMBIGUOUS);

    fillNegativeVoxelsAndAddBorder(voxelLocations, borderWidth);
    floodFillVisibleVoxels(voxelLocations, borderWidth);

    removeOutsideBorder(voxelLocations, borderWidth);

    int numberOfRecalculatedVoxels = recalculateInnerVoxels(voxelLocations, truncationThreshold);
    printMsg("Recalculated " + std::to_string(numberOfRecalculatedVoxels) + " voxels");
}
