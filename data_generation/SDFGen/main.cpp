#include <iostream>
#include <tclap/CmdLine.h>
#include "reader/SuncgReader.h"
#include "container/Array3D.h"
#include "container/Space.h"
#include "test/PolygonTest.h"
#include "util/Hdf5Writer.h"
#include "util/StopWatchPrinter.h"
#include "util/ObjWriter.h"
#include "container/PointOctree.h"
#include "geom/math/DistPoint.h"
#include "geom/math/deform.h"
#include "reader/Front3DReader.h"
#include <mutex>
#include <unistd.h>
#include <random>
#include "util/MathUtilityFunctions.h"
#include "util/CamPose.h"
#include "util/NormalCheckerRayCaster.h"
#include "reader/ReplicaReader.h"

#define MAX_AMOUNT_OF_USED_ANCHOR_POINTS 3

std::array<dPoint, 2> minMaxForPolygons(const Polygons& polygons){
    dPoint minVal = d_ones * 1000.0;
    dPoint maxVal = d_ones * -1000.0;
    for(const auto& poly: polygons){
        for(const auto& point: poly.getPoints()){
            for(int i = 0; i < 3; ++i){
                if(point[i] < minVal[i]){
                    minVal[i] = point[i];
                }
                if(point[i] > maxVal[i]){
                    maxVal[i] = point[i];
                }
            }
        }
    }
    return {minVal, maxVal};
}


void splitPolygonsIntoBlocks(PolygonReader* polygonReader, const CamPose& camPose, const std::string& outputFolder, double scaling,
                             double projectionNearClipping, double projectionFarClipping, double nearClipping, double farClipping,
                             double surfacelessPolygonsThreshold){
    Polygons polygons(readInAndPreparePolygons(camPose, *polygonReader, scaling, projectionNearClipping, projectionFarClipping,
                                     nearClipping, farClipping, surfacelessPolygonsThreshold));
    const int amountOfBlocks = 16;
    const dPoint cubeSize = d_ones * 2.0 * (1.0 / amountOfBlocks);

    std::cout << "Total amount of polygons: " << polygons.size() << std::endl;
    int counter = 0;
    for(unsigned int i = 0; i < amountOfBlocks; ++i){
        for(unsigned int j = 0; j < amountOfBlocks; ++j){
            for(unsigned int k = 0; k < amountOfBlocks; ++k){
                const dPoint lowerSize{i * cubeSize[0], j * cubeSize[1], k * cubeSize[2]};
                const dPoint upperSize{(i + 1) * cubeSize[0], (j + 1) * cubeSize[1], (k + 1) * cubeSize[2]};
                const dPoint lowerEdge = d_negOnes + lowerSize;
                const dPoint upperEdge = d_negOnes + upperSize;
                auto selectedPolygons = removePolygonsOutOfFrustum(polygons, upperEdge, lowerEdge);
                if(selectedPolygons.empty()){
                    continue;
                }
                Polygons clippedPolygons;
                for(auto& poly: selectedPolygons){
                    Points copiedPoints;
                    std::copy(poly.getPoints().begin(), poly.getPoints().end(), std::back_inserter(copiedPoints));
                    clippedPolygons.emplace_back(copiedPoints, poly.getObjectClass());
                }
                frustumClippingOnBB(clippedPolygons, lowerEdge, upperEdge);
                std::stringstream test;
                test << outputFolder << "/test_" << i << "_" << j << "_" << k << ".obj";
                auto writer = ObjWriter(test.str());
                writer.write(clippedPolygons);
            }
        }
    }
}


void convertToResVoxelGrid(PolygonReader* polygonReader, const std::vector<CamPose> &camPoses, std::string outputFolder, double scaling,
                        double projectionNearClipping, double projectionFarClipping, double nearClipping, double farClipping,
                        double surfacelessPolygonsThreshold, unsigned int spaceResolution, double minimumOctreeVoxelSize,
                        double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads, bool useExactAlgorithm,
                        int approximationAccuracy, double truncationThreshold){
    for(int camPoseIndex = 0; camPoseIndex < camPoses.size(); camPoseIndex++){
        const CamPose &camPose = camPoses[camPoseIndex];
        Polygons polygons(readInAndPreparePolygons(camPose, *polygonReader, scaling, projectionNearClipping, projectionFarClipping,
                                                   nearClipping, farClipping, surfacelessPolygonsThreshold));
        // calculate normals
        for(auto &poly: polygons){
            poly.calcNormal();
        }

        Space space({spaceResolution, spaceResolution, spaceResolution}, d_negOnes, d_ones * 2);
        if(useExactAlgorithm)
            space.calcDistsExactly(polygons, minimumOctreeVoxelSize, maxDistanceToMinPosDist, threads, truncationThreshold);
        else
            space.calcDistsApproximately(polygons, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);

        space.correctInnerVoxel(boundaryWidth, truncationThreshold);

        const auto size = space.getData().getSize();
        Array3D<float> rotatedAndFlipped(size);
        for(unsigned int i = 0; i < size[0]; ++i){
            for(unsigned int j = 0; j < size[1]; ++j){
                for(unsigned int k = 0; k < size[2]; ++k){
                    rotatedAndFlipped(i, j, size[2] - 1 - k) = space.getData()(j, size[1] - 1 - i, k);
                }
            }
        }

        // convert to 16 bit -> to save memory
        Array3D<unsigned short> compressedArray(size);
        for(unsigned int i = 0; i < size[0]; ++i){
            for(unsigned int j = 0; j < size[1]; ++j){
                for(unsigned int k = 0; k < size[2]; ++k){
                    // clipping some of the values are below the negative trunc value
                    const auto newVal = (std::max(-(float) truncationThreshold, rotatedAndFlipped(i, j, k)) + truncationThreshold) /
                                        (2 * truncationThreshold);
                    compressedArray(i, j, k) = (unsigned short) (newVal * 65535.F);
                }
            }
        }
        const std::string outputFilePath = outputFolder + "/output_" + Utility::toString(camPoseIndex) + ".hdf5";
        Hdf5Writer::writeArrayToFile(outputFilePath, compressedArray);
        printMsg("Save output file!");
    }
}

void convertCamPosesPointsLimited(int start, int end, const std::vector<CamPose> &camPoses,
                                  PolygonReader* polygonReader,
                                  std::string outputFolder, double scaling, double projectionNearClipping,
                                  double projectionFarClipping, double nearClipping,
                                  double farClipping, double surfacelessPolygonsThreshold,
                                  double minimumOctreeVoxelSize, double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads,
                                  int approximationAccuracy, double truncationThreshold, const int totalAmountOfPoints){
    for(int camPoseIndex = start; camPoseIndex < end; camPoseIndex++){
        const CamPose &camPose = camPoses[camPoseIndex];
        std::stringstream nr;
        nr << camPoseIndex;
        printMsg("Start with cam pose: " << camPoseIndex);
        // for all camera poses read in the polygons
        Polygons polygons(
                readInAndPreparePolygons(camPose, *polygonReader, scaling, projectionNearClipping, projectionFarClipping,
                                         nearClipping, farClipping, surfacelessPolygonsThreshold));
        // calculate normals
        for (auto &poly : polygons) {
            poly.calcNormal();
        }

        // check if how many points are
        auto normalRayCaster = NormalCheckerRayCaster(polygons);
        const double relativeBrokenNormalsValue = normalRayCaster.calcAmountOfBrokenNormals(512);
        // change this value for replica from 0.05 to 0.5 percent
        if(relativeBrokenNormalsValue > 0.5){ // more than 0.5 percent of the normals are broken
            std::stringstream outputFileRelativeBroken;
            outputFileRelativeBroken << outputFolder << "/" << camPoseIndex << "_relative_amount_of_broken_normals.txt";
        	std::ofstream outputRelativeNormalBroken(outputFileRelativeBroken.str(), std::ios::out);
            outputRelativeNormalBroken << relativeBrokenNormalsValue;
            outputRelativeNormalBroken.close();
            //std::stringstream outputFileRelativeImage;
            //outputFileRelativeImage << outputFolder << "/" << camPoseIndex << "_to_much_broken_normals.jpg";
            //normalRayCaster.calcNormalImageForScene(512, outputFileRelativeImage.str());
            std::cout << "Skip this camera pose as the amount of broken normals is too high." << std::endl;
            continue;
        }

        // calc small check voxelgrid with res of 128
        const unsigned int checkRes = 128;
        Space space({checkRes, checkRes, checkRes}, d_negOnes, d_ones * 2);
        space.calcDistsApproximately(polygons, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);

        space.correctInnerVoxel(boundaryWidth, truncationThreshold);

        // define a clip function, to clip points to the -1 to 1 cube
        BoundingBox bigBB({-1, -1, -1}, {1, 1, 1});
        const double anchorValue = 0.9;
        BoundingBox anchorPointsBig({-anchorValue, -anchorValue, -anchorValue}, {anchorValue, anchorValue, anchorValue});
        auto clipPoint = [bigBB](dPoint& point, const double clipValue = 0.9999){
            if(not bigBB.isPointIn(point)){
                for(int i = 0; i < 3; ++i){
                    if(point[i] < -clipValue){
                        point[i] = -clipValue;
                    }else if(point[i] > clipValue){
                        point[i] = clipValue;
                    }
                }
            }
        };

        // sample anchor points based on the visibility in the space with the checkRes
        std::vector<dPoint> anchorPoints;
        const int amountOfAnchorPoints = 400;
        Array3D<bool> visitedPlaces(space.getData().getSize());
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> uniform0To1(0, 1.0);
        visitedPlaces.fill(false);
        const auto& internalVector = space.getData().getData();
        const int internalSize = internalVector.size();
        const auto spaceSize = space.getData().getSize();
        bool skipThisCameraPose = false;
        for(unsigned int i = 0; i < amountOfAnchorPoints; ++i){
            int placeCounter = 0;
            while(true){
                const int currentIndex = (int) (uniform0To1(gen) * internalSize);

                if(internalVector[currentIndex] > 0.1 * truncationThreshold && !visitedPlaces.getData()[currentIndex]){ // free
                    visitedPlaces.getData()[currentIndex] = true;
                    const int x = currentIndex / int(spaceSize[0] * spaceSize[0]);
                    const int y = (currentIndex % int(spaceSize[0] * spaceSize[0])) / int(spaceSize[1]);
                    const int z = (currentIndex % int(spaceSize[0] * spaceSize[0])) % int(spaceSize[1]);

                    unsigned int t = visitedPlaces.getInternVal(x,y,z);
                    if(t != currentIndex){
                        assert("t is unequal the currentIndex");
                    }
                    dPoint newAnchorPoint = (dPoint(x, y, z) / checkRes) * 2 - 1;
                    // check if the anchor point is to close to the boundary
                    bool outside = false;
                    for(int j = 0; j < 3; ++j){
                        if(newAnchorPoint[j] < -0.9 || newAnchorPoint[j] > 0.9){
                            outside = true;
                            break;
                        }
                    }
                    if(!outside){
                        anchorPoints.emplace_back(std::move(newAnchorPoint));
                    }
                    break;
                }
                if(placeCounter > 1000){
                    printError("This scene does not contain enough free space to start!");
                    skipThisCameraPose = true;
                    break;
                }
                ++placeCounter;

            }
            if(skipThisCameraPose){
                break;
            }
        }
        if(skipThisCameraPose){
            continue;
        }

        Octree polygonOctree = Space::createOctree(polygons, minimumOctreeVoxelSize);
        const int totalNumberOfLeafs = polygonOctree.resetIds();

        const int amountOfPolygons = polygons.size();
        std::cout << "Amount of polygons: " << amountOfPolygons << std::endl;
        double totalPolygonSize = 0;
        for(auto &poly: polygons){
            const double size = poly.size();
            poly.calcNormal();
            totalPolygonSize += size;
        }
        std::cout << "Polygon total size: " << totalPolygonSize << std::endl;

        double stepSize = totalPolygonSize / (double) totalAmountOfPoints;
        std::vector<DistPoint> distancePoints;
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::uniform_real_distribution<> dis0to1(0, 1.0);
        distancePoints.resize(totalAmountOfPoints);
        int distIndex = 0;
        StopWatchPrinter swpPoints("Sample points");
        BoundingBox bb({0, 0, 0}, {1, 1, 1});

        auto addDistancePoint = [totalNumberOfLeafs, clipPoint, maxDistanceToMinPosDist, totalAmountOfPoints, anchorPointsBig](const Polygon &poly, std::vector<DistPoint> &distancePoints,
                                   int &distIndex, std::mt19937 &gen, const BoundingBox &bigBB, const BoundingBox &bb,
                                   double truncationThres, Space &space, double checkRes, Octree& octree, std::vector<dPoint>& anchorPoints,
                                   const bool freezeAnchor) -> bool {
            if(distIndex >= totalAmountOfPoints){
                return false;
            }
            //octree.collectLeafChilds(leafs);
            bool freeSpace = false;
            auto point = poly.getPointOnTriangle(gen);
            clipPoint(point);
            //point = point + poly.getNormal() * distance;
            std::array<DistPoint, MAX_AMOUNT_OF_USED_ANCHOR_POINTS> minValues;
            calculateMinValuesForAnchorPoints<MAX_AMOUNT_OF_USED_ANCHOR_POINTS>(minValues, anchorPoints, point);

            // check for the closest anchor points
            for(const auto& usedPoint: minValues){
                bool hitPoly = octree.rayIntersect(point, usedPoint, poly);
                if(!hitPoly){
                    distancePoints[distIndex] = DistPoint(point[0], point[1], point[2], 0.0, &poly);
                    ++distIndex;
                    if(!freezeAnchor){
                        const double distBetweenSurfaceAndAnchor = 0.025;
                        point += poly.getNormal() * distBetweenSurfaceAndAnchor;
                        //point += (usedPoint - point).normalize() * distBetweenSurfaceAndAnchor;
                        if(not anchorPointsBig.isPointIn(point)){
                            return true;
                        }

                        // check if this anchor point is to close to all existing anchor points
                        double minDist = DBL_MAX;
                        for(const dPoint &anchorPoint: anchorPoints){
                            const auto dist = (anchorPoint - point).lengthSquared();
                            if(dist < minDist){
                                minDist = dist;
                            }
                        }
                        // minDist is still squared -> square 0.05
                        if(minDist > 0.05 * 0.05){ // only if the distance to existing points is big enough
                            // check if a poly is inbetween this new point and the last anchor point
                            hitPoly = octree.rayIntersect(point, usedPoint, poly);
                            if(!hitPoly){
                                dPoint newPoint = point + poly.getNormal() * 0.01;
                                clipPoint(newPoint, 0.9);
                                bool secondHitPoly = octree.rayIntersect(newPoint, usedPoint, poly);
                                if(!secondHitPoly){
                                    std::vector<const Octree *> neighbours;
                                    std::vector<bool> visitedNodes(totalNumberOfLeafs);

                                    std::vector<QueueElement> openList;
                                    openList.reserve(totalNumberOfLeafs);
                                    const Polygon *closestPoly = nullptr;
                                    const double distance = Space::interCalcDistExactlyForPoint(point, octree, openList, neighbours,
                                                                                                visitedNodes, closestPoly,
                                                                                                maxDistanceToMinPosDist);
                                    if(closestPoly != nullptr && distance >= distBetweenSurfaceAndAnchor * 0.8){
                                        anchorPoints.emplace_back(point[0], point[1], point[2]);
                                    }
                                }
                            }
                        }
                    }
                    return true;
                }
            }
            return false;
        };


        double currentPolyCounter = 0;
        bool finalStep = false;
        bool freezeAnchor = false;
        std::vector<bool> usedPoly;
        usedPoly.resize(polygons.size());
        std::fill(usedPoly.begin(), usedPoly.end(), false);
        int oldIndex = 0;
        int currentIteration = 0;
        while(distIndex < totalAmountOfPoints || finalStep){
            unsigned int polyCounter = 0;
            if(finalStep){
                // final step with the calculated points
                distIndex = 0;
                distancePoints.clear();
                distancePoints.resize(totalAmountOfPoints);
                currentPolyCounter = 0;
                // freeze the anchor points
                freezeAnchor = true;
                finalStep = false;
                // check all not used polygons again and increase the amount of anchor points used
                StopWatchPrinter swpPoints2("Check not used polygons");
                std::vector<DistPoint> newFakePoints;
                newFakePoints.resize(polygons.size());
                int fakePointIndex = 0;
                for(const auto &poly: polygons){
                    if(!usedPoly[polyCounter]){
                        const bool useThisPoly = addDistancePoint(poly, newFakePoints, fakePointIndex, gen, bigBB, bb,
                                                                  truncationThreshold,space, checkRes, polygonOctree,
                                                                  anchorPoints, freezeAnchor);
                        usedPoly[polyCounter] = useThisPoly || usedPoly[polyCounter];
                    }
                    ++polyCounter;
                }
                swpPoints2.finish();
                // recalc the total polygon size by not using the polys which are not reachable
                double oldPolygonSize = totalPolygonSize;
                totalPolygonSize = 0;
                polyCounter = 0;
                for(const auto &poly: polygons){
                    if(!usedPoly[polyCounter]){
                        ++polyCounter;
                        continue;
                    }
                    const double size = poly.size();
                    totalPolygonSize += size;
                    ++polyCounter;
                }
                stepSize = totalPolygonSize / double(totalAmountOfPoints);
                std::cout << "The new polygon size is: " << totalPolygonSize << " down from: " << oldPolygonSize << std::endl;
            }
            polyCounter = 0;
            // to make the start more random
            currentPolyCounter = dis0to1(gen) * stepSize;
            printMsg(distIndex << ", " << totalAmountOfPoints << ", " << camPoseIndex);
            for(const auto &poly: polygons){
                if(freezeAnchor && !usedPoly[polyCounter]){
                    ++polyCounter;
                    continue;
                }
                const double size = poly.size();
                if(size > stepSize){
                    // as long as the step counter is smaller than the end point of the current
                    double currentStepCounter = currentPolyCounter;
                    for(; currentStepCounter < size + currentPolyCounter; currentStepCounter += stepSize){
                        const bool useThisPoly = addDistancePoint(poly, distancePoints, distIndex, gen, bigBB, bb, truncationThreshold, space, checkRes,
                                                                  polygonOctree, anchorPoints, freezeAnchor);
                        usedPoly[polyCounter] = useThisPoly || usedPoly[polyCounter];
                    }
                    currentPolyCounter = fmod(currentStepCounter, stepSize);
                }else{
                    // only one step
                    currentPolyCounter += size;
                    // before the final step each polygon should get the change to sample at least one point
                    if(currentPolyCounter > stepSize || (!freezeAnchor && !usedPoly[polyCounter])){
                        const bool useThisPoly = addDistancePoint(poly, distancePoints, distIndex, gen, bigBB, bb, truncationThreshold, space, checkRes,
                                                                  polygonOctree, anchorPoints, freezeAnchor);
                        usedPoly[polyCounter] = useThisPoly || usedPoly[polyCounter];
                        currentPolyCounter = 0;
                    }
                }
                ++polyCounter;
            }
            if(freezeAnchor){
                break;
            }
            oldIndex = distIndex;
            if(distIndex >= totalAmountOfPoints && !freezeAnchor){
                finalStep = true;
            }
            ++currentIteration;
        }
        distancePoints.resize(distIndex);
        swpPoints.finish();
        StopWatchPrinter swpFinalPoints("Final dist calc points");


        // scale the z axis with the squrared root to avoid making objects in the distance too small
        const bool mapPoints = true;
        if(mapPoints){
            for(auto &point : distancePoints){
                Mapping::mapPoint(point);
            }
            for(auto &point : anchorPoints){
                Mapping::mapPoint(point);
            }
        }

        const Polygon surfaceTestPolygon({{d_zeros, 0},
                                          {d_zeros, 1},
                                          {d_zeros, 2}}, -1);
        auto checkIfHitPolgyon = [&anchorPoints, &polygonOctree, &surfaceTestPolygon](const dPoint& checkPoint) -> bool {
            // calculate the sign, by checking if the point is inside or outside
            std::array<DistPoint, MAX_AMOUNT_OF_USED_ANCHOR_POINTS> minValues;
            // find the maxAmountOfUsedAnchorPoints the closest anchor points save them in minValues
            calculateMinValuesForAnchorPoints<MAX_AMOUNT_OF_USED_ANCHOR_POINTS>(minValues, anchorPoints, checkPoint);

            // check for the closest anchor points
            bool hitPoly = true;
            for(const auto &usedMinPoint: minValues){
                dPoint copyFinalPoint(checkPoint);
                dPoint copyAnchorPoint(usedMinPoint);
                // mapping back necessary as the polygons can not be mapped into the other space
                Mapping::mapPointBack(copyFinalPoint);
                Mapping::mapPointBack(copyAnchorPoint);
                hitPoly = polygonOctree.rayIntersect(copyFinalPoint, copyAnchorPoint, surfaceTestPolygon);
                if(!hitPoly){
                    // no poly was hit so this must be outside
                    break;
                }
            }
            return hitPoly;
        };
        const int amountOfOccupiedCheckVoxels = 256;
        Array3D<float> occupiedSpace({amountOfOccupiedCheckVoxels, amountOfOccupiedCheckVoxels, amountOfOccupiedCheckVoxels});

        std::normal_distribution<double> normalDis(0.0, 1.0);
        const int currentAmountOfPoints = (int) distancePoints.size();
        const int finalAmountOfPoints = 2000000;
        std::vector<DistPoint> finalPoints;
        finalPoints.resize(finalAmountOfPoints);
        const int maxAmountOfUsedAnchorPoints = 3;
        if(mapPoints){
            // set the amount of splits used for the point octree
            const int pointMaxLevel = 4;
            std::vector<DistPoint> distancePointsCopy(distancePoints);
            PointOctree pointOctree(std::move(distancePointsCopy), pointMaxLevel, {2, 2, 2}, {-1, -1, -1});
            for(int i = 0; i < finalAmountOfPoints; ++i){
                bool foundAPoint = false;
                while(!foundAPoint){
                    const int currentSampledIndex = int(dis0to1(gen) * currentAmountOfPoints);
                    const auto &usedPoint = distancePoints[currentSampledIndex];

                    dPoint dir;
                    double totalLen = 0.0;
                    for(int j = 0; j < 3; ++j){
                        dir[j] = normalDis(gen);
                        totalLen += dir[j] * dir[j];
                    }
                    dir /= sqrt(totalLen);

                    dir *= dis0to1(gen) * truncationThreshold * 2.0;
                    const dPoint finalPoint = usedPoint + dir;
                    if(bigBB.isPointIn(finalPoint)){
                        // calc the distance
                        PointDist minDist = pointOctree.distance(finalPoint);
                        // check if a class was found
                        if(minDist.second != nullptr){
                            // check if the distance is closer than 115 % of the truncation threshold
                            if(abs(minDist.first) < truncationThreshold * 1.15){
                                // calculate hit point:
                                const dPoint& hitPointCurvedTriangle = *minDist.second;
                                // improve the value
                                double newSmallerDist = minDist.second->getPolygon()->sampleCloserPoints(hitPointCurvedTriangle, finalPoint, gen);
                                if(newSmallerDist < minDist.first){
                                    const double improveDiff = minDist.first - newSmallerDist;
                                    minDist.first = newSmallerDist;
                                }
                            }

                            const bool hitPoly = checkIfHitPolgyon(finalPoint);
                            if(hitPoly){
                                minDist.first *= -1;
                            }
                            const double usedDist = std::min(truncationThreshold, std::max(-truncationThreshold, minDist.first));
                            finalPoints[i] = DistPoint(finalPoint[0], finalPoint[1], finalPoint[2], usedDist, minDist.second->getPolygon());
                            foundAPoint = true;
                        }else{
                            printError("There was a point, where not class was found: " << finalPoint << ", res: " << minDist.first << ", " << minDist.second);
                        }
                    }
                }
            }
            StopWatchPrinter swpOccupancyGrid2("Calc occupancy grid point octree");

            std::vector<DistPoint> finalPointsCopy(finalPoints);
            PointOctree finalPointsOctree(std::move(finalPointsCopy), pointMaxLevel, {2, 2, 2}, {-1, -1, -1});
            const int pointOctreeRes = int(pow(2, (pointMaxLevel + 1)));
            const int valuesPerVoxel = amountOfOccupiedCheckVoxels / pointOctreeRes;

            for(int i = 0; i < pointOctreeRes; ++i){
                for(int j = 0; j < pointOctreeRes; ++j){
                    for(int k = 0; k < pointOctreeRes; ++k){
                        dPoint outerPoint(i, j, k);
                        outerPoint += 0.5;
                        outerPoint /= pointOctreeRes * 0.5;
                        outerPoint -= 1.0;
                        if(finalPointsOctree.findNodeContainingPoint(outerPoint).getState() == OCTREE_EMPTY){
                            // if empty, check if positive or negative
                            // we can do this check here as there are no points in this octree, so we just check if this a complete
                            // empty or full voxel
                            const bool hitPoly = checkIfHitPolgyon(outerPoint);
                            // true if a polygon was hit -> that means inside
                            const float usedValue = float(hitPoly ? -truncationThreshold: truncationThreshold);
                            for(int l = 0; l < valuesPerVoxel; ++l){
                                for(int m = 0; m < valuesPerVoxel; ++m){
                                    for(int n = 0; n < valuesPerVoxel; ++n){
                                        occupiedSpace(i*valuesPerVoxel+l, j*valuesPerVoxel+m, k*valuesPerVoxel+n) = usedValue;
                                    }
                                }
                            }
                        }else{
                            for(int l = 0; l < valuesPerVoxel; ++l){
                                for(int m = 0; m < valuesPerVoxel; ++m){
                                    for(int n = 0; n < valuesPerVoxel; ++n){
                                        dPoint finalPoint(i*valuesPerVoxel+l, j*valuesPerVoxel+m, k*valuesPerVoxel+n);
                                        finalPoint += 0.5;
                                        finalPoint /= amountOfOccupiedCheckVoxels * 0.5;
                                        finalPoint -= 1.0;
                                        // calc the distance
                                        PointDist minDist = pointOctree.distance(finalPoint);
                                        // check if a class was found
                                        const bool hitPoly = checkIfHitPolgyon(finalPoint);
                                        if(minDist.second != nullptr){
                                            // check if the distance is closer than 115 % of the truncation threshold
                                            //if(abs(minDist.first) < truncationThreshold * 1.15){
                                            //    // calculate hit point:
                                            //    const dPoint &hitPointCurvedTriangle = *minDist.second;
                                            //    // improve the value
                                            //    double newSmallerDist = minDist.second->getPolygon()->sampleCloserPoints(
                                            //            hitPointCurvedTriangle,
                                            //            finalPoint, gen);
                                            //    if(newSmallerDist < minDist.first){
                                            //        const double improveDiff = minDist.first - newSmallerDist;
                                            //        minDist.first = newSmallerDist;
                                            //    }
                                            //}

                                            if(hitPoly){
                                                minDist.first *= -1;
                                            }
                                            const double usedDist = std::min(truncationThreshold, std::max(-truncationThreshold, minDist.first));
                                            occupiedSpace(i*valuesPerVoxel+l, j*valuesPerVoxel+m, k*valuesPerVoxel+n) = float(usedDist);
                                        }else{
                                            // true if a polygon was hit -> that means inside
                                            if(hitPoly){
                                                occupiedSpace(i*valuesPerVoxel+l, j*valuesPerVoxel+m, k*valuesPerVoxel+n) = -truncationThreshold;
                                            }else{
                                                occupiedSpace(i*valuesPerVoxel+l, j*valuesPerVoxel+m, k*valuesPerVoxel+n) = truncationThreshold;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            swpOccupancyGrid2.finish();
        }else{

            Octree bigPolygonOctree = Space::createOctree(polygons, 2 * truncationThreshold);

            // if no projection is used the distance calculation can be done for each point towards each polygon
            for(int i = 0; i < finalAmountOfPoints; ++i){
                bool foundAPoint = false;
                while(!foundAPoint){
                    const int currentSampledIndex = int(dis0to1(gen) * currentAmountOfPoints);
                    const auto &usedPoint = distancePoints[currentSampledIndex];

                    dPoint dir;
                    double totalLen = 0.0;
                    for(int j = 0; j < 3; ++j){
                        dir[j] = normalDis(gen);
                        totalLen += dir[j] * dir[j];
                    }
                    dir /= sqrt(totalLen);

                    dir *= dis0to1(gen) * truncationThreshold * 2.0;
                    const dPoint finalPoint = usedPoint + dir;
                    if(bigBB.isPointIn(finalPoint)){
                        double minDist = DBL_MAX;
                        Polygon* const* polyPointer = nullptr;
                        const Octree& currentOctree = bigPolygonOctree.findNodeContainingPoint(finalPoint);
                        for(const auto& poly: currentOctree.getIntersectingPolygons()){
                            const double newDist = std::abs(poly->calcDistanceConst(finalPoint));
                            if(newDist < minDist){
                                minDist = newDist;
                                polyPointer = &poly;
                            }
                        }
                        std::vector<const Octree*> neighbors;
                        currentOctree.findLeafNeighbours(neighbors);
                        for(const auto& neighborOctree: neighbors){
                            if(neighborOctree != nullptr){
                                for(const auto &poly: neighborOctree->getIntersectingPolygons()){
                                    const double newDist = std::abs(poly->calcDistanceConst(finalPoint));
                                    if(newDist < minDist){
                                        minDist = newDist;
                                        polyPointer = &poly;
                                    }
                                }
                            }
                        }

                        // calculate the sign, by checking if the point is inside or outside
                        std::array<DistPoint, MAX_AMOUNT_OF_USED_ANCHOR_POINTS> minValues;
                        // find the maxAmountOfUsedAnchorPoints closest anchor points in minValues
                        calculateMinValuesForAnchorPoints<MAX_AMOUNT_OF_USED_ANCHOR_POINTS>(minValues, anchorPoints, finalPoint);

                        // check for the closest anchor points
                        bool hitPoly = true;
                        for(const auto &usedPoint: minValues){
                            hitPoly = polygonOctree.rayIntersect(finalPoint, usedPoint, surfaceTestPolygon);
                            if(!hitPoly){
                                // no poly was hit so this must be outside
                                break;
                            }
                        }

                        double usedDist = std::min(truncationThreshold, minDist);
                        if(hitPoly){
                            usedDist *= -1;
                        }
                        if(polyPointer != nullptr){
                            finalPoints[i] = DistPoint(finalPoint[0], finalPoint[1], finalPoint[2], usedDist,
                                                       reinterpret_cast<const Polygon *>(polyPointer));
                            foundAPoint = true;
                        }else{
                            printMsg("Point not inside" << finalPoint);
                        }
                    }
                }
            }
        }
        swpFinalPoints.finish();

        std::stringstream str;
        str << "Writing " << camPoseIndex << ".hdf5";
        StopWatchPrinter swp1(str.str());
        Hdf5Writer::hdf5_writer_mutex.lock();
        hid_t fileId = Hdf5Writer::openFile(outputFolder + "/" + "result_" + nr.str() + ".hdf5");
        //Hdf5Writer::writeArrayToFile(fileId, finalSpace);

        std::vector<std::array<double, 3> > points;
        points.resize(finalPoints.size());
        std::vector<std::array<double, 1> > dists;
        dists.resize(finalPoints.size());
        std::vector<std::array<unsigned char, 1> > classes;
        classes.resize(finalPoints.size());
        for(int i = 0; i < finalPoints.size(); ++i){
            for(int j = 0; j < 3; ++j){
                points[i][j] = finalPoints[i][j];
            }
            dists[i][0] = finalPoints[i].getDist();
            classes[i][0] = finalPoints[i].getClass();
        }

        Hdf5Writer::writeVectorToFileId(fileId, "points", points);
        Hdf5Writer::writeVectorToFileId(fileId, "distances", dists);
        Hdf5Writer::writeVectorToFileId(fileId, "classes", classes);
        Hdf5Writer::writeContainerOfPointsToFile(fileId, "anchor_points", anchorPoints);
        std::stringstream camera_number;
        camera_number << "Current camera nr: ";
        camera_number << camPoseIndex;
        {
            Array3D<unsigned short> compressedArray(occupiedSpace.getSize());
            for(unsigned int i = 0; i < compressedArray.getSize()[0]; ++i){
                for(unsigned int j = 0; j < compressedArray.getSize()[1]; ++j){
                    for(unsigned int k = 0; k < compressedArray.getSize()[2]; ++k){
                        // clipping some of the values are below the negative trunc value
                        const auto newVal = occupiedSpace(i,j,k) / (2 * truncationThreshold) + 0.5;
                        if(newVal > 1.0){
                            compressedArray(i, j, k) = 65535;
                        }else if(newVal < 0.0){
                            compressedArray(i, j, k) = 0;
                        }else{
                            compressedArray(i, j, k) = (unsigned short) (newVal * 65535.F);
                        }
                    }
                }
            }
            Hdf5Writer::writeArrayToFileId(fileId, "voxel_space", compressedArray);
        }
        Hdf5Writer::writeStringToFileId(fileId, "camera_number", camera_number.str());
        polygonReader->addInfoToHdf5Container(fileId);
        Hdf5Writer::writeStringToFileId(fileId, "camera_pose", convert_to_string(camPose));
        Hdf5Writer::writeStringToFileId(fileId, "used_params", Utility::convert_to_string_as_yaml(checkRes,  scaling,  projectionNearClipping,
                                   projectionFarClipping,  nearClipping,
                                   farClipping,  surfacelessPolygonsThreshold,
                                  minimumOctreeVoxelSize,
                                   maxDistanceToMinPosDist,  boundaryWidth, threads,
                                  approximationAccuracy,  truncationThreshold, mapPoints, relativeBrokenNormalsValue));
        H5Fclose(fileId);
        Hdf5Writer::hdf5_writer_mutex.unlock();
        swp1.finish();


    }
}

int front3d_block_the_file(int argc, char **argv){
    TCLAP::CmdLine cmd("Generate blocks passed on a list of camera positions and an .obj file", ' ', "1.0");
    const bool required = true;
    const bool notRequired = false;
    TCLAP::ValueArg<std::string> wallObjFile("w", "wall_obj", "File path to the wall obj file", required, "", "string");
    TCLAP::ValueArg<std::string> objPositionText("p", "position_txt", "File path to the position txt file", required, "", "string");
    TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file",
                                                     required, "", "string");
    TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
    TCLAP::ValueArg<int> cameraNr("n", "camera_nr", "Used camera number.", required, 0, "int");
    TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
    TCLAP::ValueArg<double> totalAmountOfPoints("", "amountOfPoints", "Total amount of points sampled", notRequired, 1000000, "double");
    TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons",
                                        notRequired, 7, "double");
    TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons",
                                         notRequired, 1, "double");
    TCLAP::ValueArg<double> projectionFarClipping("", "proj_far",
                                                  "Far clipping threshold used for building the projection matrix",
                                                  notRequired, 4, "double");
    TCLAP::ValueArg<double> projectionNearClipping("", "proj_near",
                                                   "Near clipping threshold used for building the projection matrix",
                                                   notRequired, 1, "double");
    TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres",
                                                         "Threshold for detection of polygons with no surface",
                                                         notRequired, 1e-4, "double");

    cmd.add(wallObjFile);
    cmd.add(objPositionText);
    cmd.add(cameraPositionsFile);
    cmd.add(outputFolder);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(cameraNr);
    cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.parse(argc, argv);

    std::vector<CamPose> camPoses = readCameraPoses(cameraPositionsFile.getValue());
    Front3DReader front3DReader(objPositionText.getValue(), wallObjFile.getValue());
    splitPolygonsIntoBlocks(&front3DReader, camPoses[cameraNr.getValue()], outputFolder.getValue(), scaling.getValue() , projectionNearClipping.getValue(),
                            projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(),
                            surfacelessPolygonsThreshold.getValue());
    std::cout << "Done" << std::endl;
    return 0;
}

int replica_main(int argc, char **argv){
    //PolygonTest::testAll();
    TCLAP::CmdLine cmd("Generate Voxels passed on a list of camera postions and an .obj file", ' ', "1.0");
    const bool required = true;
    const bool notRequired = false;
    TCLAP::ValueArg<std::string> objFile("o", "object_path", "File path to the wall obj file", required, "", "string");
    TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file",
                                                     required, "", "string");
    TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
    TCLAP::ValueArg<double> minimumOctreeVoxelSize("d", "depth",
                                                   "Minimum octree voxel size. Will be used to determine octree depth.",
                                                   notRequired, 0.0625, "double");
    TCLAP::ValueArg<int> boundaryWidth("b", "boundary", "Additional boundary width for inner voxel detection",
                                       notRequired, 2, "int");
    TCLAP::ValueArg<int> cameraNr("", "camera_nr", "If a camera nr is given only that camera id is calculated.", notRequired, -1, "int");
    TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres",
                                                         "Threshold for detection of polygons with no surface",
                                                         notRequired, 1e-4, "double");
    TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
    TCLAP::ValueArg<double> totalAmountOfPoints("", "amountOfPoints", "Total amount of points sampled", notRequired, 1000000, "double");
    TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons",
                                        notRequired, 7, "double");
    TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons",
                                         notRequired, 1, "double");
    TCLAP::ValueArg<double> projectionFarClipping("", "proj_far",
                                                  "Far clipping threshold used for building the projection matrix",
                                                  notRequired, 4, "double");
    TCLAP::ValueArg<double> projectionNearClipping("", "proj_near",
                                                   "Near clipping threshold used for building the projection matrix",
                                                   notRequired, 1, "double");
    TCLAP::ValueArg<double> frustumBorder("", "frustum_bor", "Additional border around frustum when clipping polygons",
                                          notRequired, 0, "double");
    TCLAP::ValueArg<double> maxDistanceToMinPosDist("", "pos_threshold",
                                                    "Maximum amount the minimum positive distance is allowed to be smaller than the minimum negative distance to a polygon, s.t. the voxel should be still positive.",
                                                    notRequired, 4e-3, "double");
    TCLAP::ValueArg<unsigned int> threads("", "threads", "Number of threads to use, ==0 uses the maximum of threads possible, smaller than 0 uses only 1.", notRequired, 8, "int");
    TCLAP::ValueArg<int> approximationAccuracy("", "accuracy",
                                               "When using the approximate algorithm, this determines the area around each voxel which is searched.",
                                               notRequired, 2, "int");
    TCLAP::ValueArg<double> truncationThreshold("", "trunc", "The truncation threshold to use", notRequired, 0.1,
                                                "double");


    cmd.add(objFile);
    cmd.add(cameraPositionsFile);
    cmd.add(outputFolder);
    cmd.add(minimumOctreeVoxelSize);
    cmd.add(boundaryWidth);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.add(frustumBorder);
    cmd.add(maxDistanceToMinPosDist);
    cmd.add(threads);
    cmd.add(cameraNr);
    cmd.add(approximationAccuracy);
    cmd.add(truncationThreshold);
    cmd.add(totalAmountOfPoints);
    cmd.parse(argc, argv);

    std::vector<CamPose> camPoses = readCameraPoses(cameraPositionsFile.getValue());
    if(cameraNr.getValue() != -1){
        if(cameraNr.getValue() >= camPoses.size()){
            printError("The selected cam file has only " << camPoses.size() << " cam poses, could not access nr: " << cameraNr.getValue());
            exit(1);
        }
        const auto savedCamPose = camPoses[cameraNr.getValue()];
        camPoses.clear();
        camPoses.emplace_back(savedCamPose);
    }
    ReplicaReader replicaReader(objFile.getValue());

    StopWatchPrinter mainSwp("Total run time for " + std::to_string(camPoses.size()));
    unsigned int amountOfThreads = threads.getValue();
    if(amountOfThreads < 0){
        convertCamPosesPointsLimited(0, camPoses.size(), camPoses, &replicaReader, outputFolder.getValue(), scaling.getValue(),
                        projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(),
                        farClipping.getValue(), surfacelessPolygonsThreshold.getValue(),
                        minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(),
                        threads.getValue(), approximationAccuracy.getValue(),
                        truncationThreshold.getValue(), totalAmountOfPoints.getValue());
    }else{
        if(amountOfThreads == 0)
            amountOfThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        amountOfThreads = std::min(amountOfThreads, (unsigned int) camPoses.size());
        for(unsigned int i = 0; i < amountOfThreads; ++i){
            auto start = (unsigned int) (i * camPoses.size() / (float) amountOfThreads);
            auto end = (unsigned int) ((i + 1) * camPoses.size() / (float) amountOfThreads);
            if(i + 1 == amountOfThreads){
                end = camPoses.size();
            }
            threads.emplace_back(
                    std::thread(&convertCamPosesPointsLimited, start, end, std::cref(camPoses), &replicaReader,
                                outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(),
                                projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(),
                                surfacelessPolygonsThreshold.getValue(),
                                minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(),
                                boundaryWidth.getValue(), 1,
                                approximationAccuracy.getValue(), truncationThreshold.getValue(), totalAmountOfPoints.getValue())
            );
        }
        for(auto &thread : threads){
            thread.join();
        }
    }
    mainSwp.finish();

    std::cout << "Done" << std::endl;
    return 0;
}



int front3d_main(int argc, char **argv){
    //PolygonTest::testAll();
    TCLAP::CmdLine cmd("Generate Voxels passed on a list of camera postions and an .obj file", ' ', "1.0");
    const bool required = true;
    const bool notRequired = false;
    TCLAP::ValueArg<std::string> wallObjFile("w", "wall_obj", "File path to the wall obj file", required, "", "string");
    TCLAP::ValueArg<std::string> objPositionText("p", "position_txt", "File path to the position txt file", required, "", "string");
    TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file",
                                                     required, "", "string");
    TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
    TCLAP::ValueArg<double> minimumOctreeVoxelSize("d", "depth",
                                                   "Minimum octree voxel size. Will be used to determine octree depth.",
                                                   notRequired, 0.0625, "double");
    TCLAP::ValueArg<int> boundaryWidth("b", "boundary", "Additional boundary width for inner voxel detection",
                                       notRequired, 2, "int");
    TCLAP::ValueArg<int> cameraNr("", "camera_nr", "If a camera nr is given only that camera id is calculated.", notRequired, -1, "int");
    TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres",
                                                         "Threshold for detection of polygons with no surface",
                                                         notRequired, 1e-4, "double");
    TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
    TCLAP::ValueArg<double> totalAmountOfPoints("", "amountOfPoints", "Total amount of points sampled", notRequired, 1000000, "double");
    TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons",
                                        notRequired, 7, "double");
    TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons",
                                         notRequired, 1, "double");
    TCLAP::ValueArg<double> projectionFarClipping("", "proj_far",
                                                  "Far clipping threshold used for building the projection matrix",
                                                  notRequired, 4, "double");
    TCLAP::ValueArg<double> projectionNearClipping("", "proj_near",
                                                   "Near clipping threshold used for building the projection matrix",
                                                   notRequired, 1, "double");
    TCLAP::ValueArg<double> frustumBorder("", "frustum_bor", "Additional border around frustum when clipping polygons",
                                          notRequired, 0, "double");
    TCLAP::ValueArg<double> maxDistanceToMinPosDist("", "pos_threshold",
                                                    "Maximum amount the minimum positive distance is allowed to be smaller than the minimum negative distance to a polygon, s.t. the voxel should be still positive.",
                                                    notRequired, 4e-3, "double");
    TCLAP::ValueArg<unsigned int> threads("", "threads", "Number of threads to use, ==0 uses the maximum of threads possible, smaller than 0 uses only 1.", notRequired, 8, "int");
    TCLAP::ValueArg<int> approximationAccuracy("", "accuracy",
                                               "When using the approximate algorithm, this determines the area around each voxel which is searched.",
                                               notRequired, 2, "int");
    TCLAP::ValueArg<double> truncationThreshold("", "trunc", "The truncation threshold to use", notRequired, 0.1,
                                                "double");


    cmd.add(wallObjFile);
    cmd.add(objPositionText);
    cmd.add(cameraPositionsFile);
    cmd.add(outputFolder);
    cmd.add(minimumOctreeVoxelSize);
    cmd.add(boundaryWidth);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.add(frustumBorder);
    cmd.add(maxDistanceToMinPosDist);
    cmd.add(threads);
    cmd.add(cameraNr);
    cmd.add(approximationAccuracy);
    cmd.add(truncationThreshold);
    cmd.add(totalAmountOfPoints);
    cmd.parse(argc, argv);

    std::vector<CamPose> camPoses = readCameraPoses(cameraPositionsFile.getValue());
    if(cameraNr.getValue() != -1){
        if(cameraNr.getValue() >= camPoses.size()){
            printError("The selected cam file has only " << camPoses.size() << " cam poses, could not access nr: " << cameraNr.getValue());
            exit(1);
        }
        const auto savedCamPose = camPoses[cameraNr.getValue()];
        camPoses.clear();
        camPoses.emplace_back(savedCamPose);
    }
    Front3DReader front3DReader(objPositionText.getValue(), wallObjFile.getValue());

    StopWatchPrinter mainSwp("Total run time for " + std::to_string(camPoses.size()));
    unsigned int amountOfThreads = threads.getValue();
    if(amountOfThreads < 0){
        convertCamPosesPointsLimited(0, camPoses.size(), camPoses, &front3DReader, outputFolder.getValue(), scaling.getValue(),
                        projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(),
                        farClipping.getValue(), surfacelessPolygonsThreshold.getValue(),
                        minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(),
                        threads.getValue(), approximationAccuracy.getValue(),
                        truncationThreshold.getValue(), totalAmountOfPoints.getValue());
    }else{
        if(amountOfThreads == 0)
            amountOfThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        amountOfThreads = std::min(amountOfThreads, (unsigned int) camPoses.size());
        for(unsigned int i = 0; i < amountOfThreads; ++i){
            auto start = (unsigned int) (i * camPoses.size() / (float) amountOfThreads);
            auto end = (unsigned int) ((i + 1) * camPoses.size() / (float) amountOfThreads);
            if(i + 1 == amountOfThreads){
                end = camPoses.size();
            }
            threads.emplace_back(
                    std::thread(&convertCamPosesPointsLimited, start, end, std::cref(camPoses), &front3DReader,
                                outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(),
                                projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(),
                                surfacelessPolygonsThreshold.getValue(),
                                minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(),
                                boundaryWidth.getValue(), 1,
                                approximationAccuracy.getValue(), truncationThreshold.getValue(), totalAmountOfPoints.getValue())
            );
        }
        for(auto &thread : threads){
            thread.join();
        }
    }
    mainSwp.finish();

    std::cout << "Done" << std::endl;
    return 0;
}

int front_3d_voxelgrid_main(int argc, char **argv){
    //PolygonTest::testAll();
	TCLAP::CmdLine cmd("Generate Voxels passed on a list of camera postions and an .obj file", ' ', "1.0");
	const bool required = true;
	const bool notRequired = false;
    TCLAP::ValueArg<std::string> wallObjFile("w", "wall_obj", "File path to the wall obj file", required, "", "string");
    TCLAP::ValueArg<std::string> objPositionText("p", "position_txt", "File path to the position txt file", required, "", "string");
	TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file", required, "", "string");
	TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
	TCLAP::ValueArg<double> minimumOctreeVoxelSize("d", "depth", "Minimum octree voxel size. Will be used to determie octree depth.", notRequired, 0.1, "double");
	TCLAP::ValueArg<int> boundaryWidth("b", "boundary", "Additional boundary width for inner voxel detection", notRequired, 2, "int");
	TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres", "Threshold for detection of polygons with no surface", notRequired, 1e-4, "double");
	TCLAP::ValueArg<unsigned int> spaceResolution("r", "res", "Resolution of the voxel space", notRequired, 512, "int");
	TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
	TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons", notRequired, 4, "double");
	TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons", notRequired, 1, "double");
	TCLAP::ValueArg<double> projectionFarClipping("", "proj_far", "Far clipping threshold used for building the projection matrix", notRequired, 4, "double");
	TCLAP::ValueArg<double> projectionNearClipping("", "proj_near", "Near clipping threshold used for building the projection matrix", notRequired, 1, "double");
	TCLAP::ValueArg<double> frustumBorder("", "frustum_bor", "Additional border around frustum when clipping polygons", notRequired, 0, "double");
	TCLAP::ValueArg<double> maxDistanceToMinPosDist("", "pos_threshold", "Maximum amount the minimum positive distance is allowed to be smaller than the minimum negative distance to a polygon, s.t. the voxel should be still positive.", notRequired, 4e-3, "double");
	TCLAP::ValueArg<unsigned int> threads("", "threads", "Number of threads to use", notRequired, 0, "int");
	TCLAP::ValueArg<bool> useExactAlgorithm("", "exact", "Use exact algorithm", notRequired, false, "bool");
	TCLAP::ValueArg<int> approximationAccuracy("", "accuracy", "When using the approximate algorithm, this determines the area around each voxel which is searched.", notRequired, 2, "int");
	TCLAP::ValueArg<double> truncationThreshold("", "trunc", "The truncation threshold to use", notRequired, 0.1, "double");

	cmd.add(wallObjFile);
    cmd.add(objPositionText);
	cmd.add(cameraPositionsFile);
	cmd.add(outputFolder);
    cmd.add(minimumOctreeVoxelSize);
    cmd.add(boundaryWidth);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(spaceResolution);
	cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.add(frustumBorder);
    cmd.add(maxDistanceToMinPosDist);
	cmd.add(threads);
	cmd.add(useExactAlgorithm);
	cmd.add(approximationAccuracy);
	cmd.add(truncationThreshold);
	cmd.parse(argc, argv);

    std::vector<CamPose> camPoses = readCameraPoses(cameraPositionsFile.getValue());

    Front3DReader front3DReader(objPositionText.getValue(), wallObjFile.getValue());
    convertToResVoxelGrid(&front3DReader, camPoses, outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(), surfacelessPolygonsThreshold.getValue(), spaceResolution.getValue(), minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(), 1, useExactAlgorithm.getValue(), approximationAccuracy.getValue(), truncationThreshold.getValue());
}

int sdf_main(int argc, char **argv){
    //PolygonTest::testAll();
    TCLAP::CmdLine cmd("Generate Voxels passed on a list of camera positions and an .obj file", ' ', "1.0");
    const bool required = true;
    const bool notRequired = false;
    TCLAP::ValueArg<std::string> objFile("o", "obj", "File path to the house json file", required, "", "string");
    TCLAP::ValueArg<std::string> cameraPositionsFile("c", "cameraPosFile", "File path to camera position file",
                                                     required, "", "string");
    TCLAP::ValueArg<std::string> outputFolder("f", "folder", "Folder path for output files", required, "", "string");
    TCLAP::ValueArg<double> minimumOctreeVoxelSize("d", "depth",
                                                   "Minimum octree voxel size. Will be used to determine octree depth.",
                                                   notRequired, 0.0625, "double");
    TCLAP::ValueArg<int> boundaryWidth("b", "boundary", "Additional boundary width for inner voxel detection",
                                       notRequired, 2, "int");
    TCLAP::ValueArg<double> surfacelessPolygonsThreshold("t", "thres",
                                                         "Threshold for detection of polygons with no surface",
                                                         notRequired, 1e-4, "double");
    TCLAP::ValueArg<double> scaling("s", "scale", "Polygon scaling", notRequired, 1, "double");
    TCLAP::ValueArg<double> totalAmountOfPoints("", "amountOfPoints", "Total amount of points sampled", notRequired, 1000000, "double");
    TCLAP::ValueArg<double> farClipping("", "far", "Far clipping threshold used for removing outside polygons",
                                        notRequired, 7, "double");
    TCLAP::ValueArg<double> nearClipping("", "near", "Near clipping threshold used for removing outside polygons",
                                         notRequired, 1, "double");
    TCLAP::ValueArg<double> projectionFarClipping("", "proj_far",
                                                  "Far clipping threshold used for building the projection matrix",
                                                  notRequired, 4, "double");
    TCLAP::ValueArg<double> projectionNearClipping("", "proj_near",
                                                   "Near clipping threshold used for building the projection matrix",
                                                   notRequired, 1, "double");
    TCLAP::ValueArg<double> frustumBorder("", "frustum_bor", "Additional border around frustum when clipping polygons",
                                          notRequired, 0, "double");
    TCLAP::ValueArg<double> maxDistanceToMinPosDist("", "pos_threshold",
                                                    "Maximum amount the minimum positive distance is allowed to be smaller than the minimum negative distance to a polygon, s.t. the voxel should be still positive.",
                                                    notRequired, 4e-3, "double");
    TCLAP::ValueArg<unsigned int> threads("", "threads", "Number of threads to use, ==0 uses the maximum of threads possible, smaller than 0 uses only 1.", notRequired, 8, "int");
    TCLAP::ValueArg<int> approximationAccuracy("", "accuracy",
                                               "When using the approximate algorithm, this determines the area around each voxel which is searched.",
                                               notRequired, 2, "int");
    TCLAP::ValueArg<double> truncationThreshold("", "trunc", "The truncation threshold to use", notRequired, 0.1,
                                                "double");

    cmd.add(objFile);
    cmd.add(cameraPositionsFile);
    cmd.add(outputFolder);
    cmd.add(minimumOctreeVoxelSize);
    cmd.add(boundaryWidth);
    cmd.add(surfacelessPolygonsThreshold);
    cmd.add(scaling);
    cmd.add(farClipping);
    cmd.add(nearClipping);
    cmd.add(projectionFarClipping);
    cmd.add(projectionNearClipping);
    cmd.add(frustumBorder);
    cmd.add(maxDistanceToMinPosDist);
    cmd.add(threads);
    cmd.add(approximationAccuracy);
    cmd.add(truncationThreshold);
    cmd.add(totalAmountOfPoints);
    cmd.parse(argc, argv);

    std::vector<CamPose> camPoses = readCameraPoses(cameraPositionsFile.getValue());
    //camPoses[0] = camPoses[2];
    //camPoses.resize(1);
    std::string suncgPath, modelCategoryMappingPath, csvNYUPath;
    suncgPath = "/home/max/data/version_1.1.0";
    modelCategoryMappingPath = "/home/max/workspace/SDFGen/resources/ModelCategoryMapping.csv";
    csvNYUPath = "/home/max/workspace/SDFGen/resources/nyu_idset.csv";
    SuncgLoader suncgLoader(suncgPath, modelCategoryMappingPath, csvNYUPath, objFile.getValue());


    StopWatchPrinter mainSwp("Total run time for " + std::to_string(camPoses.size()));
    unsigned int amountOfThreads = threads.getValue();
    if(amountOfThreads < 0){
        convertCamPosesPointsLimited(0, camPoses.size(), camPoses, &suncgLoader, outputFolder.getValue(), scaling.getValue(),
                        projectionNearClipping.getValue(), projectionFarClipping.getValue(), nearClipping.getValue(),
                        farClipping.getValue(), surfacelessPolygonsThreshold.getValue(),
                        minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(), boundaryWidth.getValue(),
                        threads.getValue(), approximationAccuracy.getValue(),
                        truncationThreshold.getValue(), totalAmountOfPoints.getValue());
    }else{
        if(amountOfThreads == 0)
            amountOfThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;

        amountOfThreads = std::min(amountOfThreads, (unsigned int) camPoses.size());
        for(unsigned int i = 0; i < amountOfThreads; ++i){
            auto start = (unsigned int) (i * camPoses.size() / (float) amountOfThreads);
            auto end = (unsigned int) ((i + 1) * camPoses.size() / (float) amountOfThreads);
            if(i + 1 == amountOfThreads){
                end = camPoses.size();
            }
            threads.emplace_back(
                    std::thread(&convertCamPosesPointsLimited, start, end, std::cref(camPoses), &suncgLoader,
                                outputFolder.getValue(), scaling.getValue(), projectionNearClipping.getValue(),
                                projectionFarClipping.getValue(), nearClipping.getValue(), farClipping.getValue(),
                                surfacelessPolygonsThreshold.getValue(),
                                minimumOctreeVoxelSize.getValue(), maxDistanceToMinPosDist.getValue(),
                                boundaryWidth.getValue(), 1,
                                approximationAccuracy.getValue(), truncationThreshold.getValue(), totalAmountOfPoints.getValue())
            );
        }
        for(auto &thread : threads){
            thread.join();
        }
    }
    mainSwp.finish();

    std::cout << "Done" << std::endl;
    return 0;
}

int main(int argc, char **argv){
    return front3d_main(argc, argv);
    //return replica_main(argc, argv);
}
