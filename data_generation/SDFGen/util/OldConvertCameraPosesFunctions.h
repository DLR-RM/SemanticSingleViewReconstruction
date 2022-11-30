//
// Created by max on 04.10.21.
//

#ifndef SDFGEN_OLDCONVERTCAMERAPOSESFUNCTIONS_H
#define SDFGEN_OLDCONVERTCAMERAPOSESFUNCTIONS_H

void convertCamPosesPoints(int start, int end, const std::vector<CamPose> &camPoses, PolygonReader& polygonReader,
                           std::string outputFolder, double scaling, double projectionNearClipping,
                           double projectionFarClipping, double nearClipping,
                           double farClipping, double surfacelessPolygonsThreshold, const unsigned int spaceResolution,
                           double minimumOctreeVoxelSize,
                           double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads, bool useExactAlgorithm,
                           int approximationAccuracy, double truncationThreshold){
    for(int camPoseIndex = start; camPoseIndex < end; camPoseIndex++){
        const CamPose &camPose = camPoses[camPoseIndex];
        std::stringstream nr;
        nr << camPoseIndex;
        Polygons polygons(
                readInAndPreparePolygons(camPose, polygonReader, scaling, projectionNearClipping, projectionFarClipping,
                                         nearClipping, farClipping, surfacelessPolygonsThreshold));

        //writeToDisc(polygons, outputFolder + "/" + "result_" + nr.str() + ".obj");

        const int amountOfPolygons = polygons.size();
        std::cout << "Amount of polygons: " << amountOfPolygons << std::endl;
        double totalPolygonSize = 0;
        for(auto &poly: polygons){
            const double size = poly.size();
            poly.calcNormal();
            totalPolygonSize += size;
        }
        std::cout << "Polygon total size: " << totalPolygonSize << std::endl;
        const int maxAmount = 500000;
        const int totalAmountOfPoints = 5000000;
        const double stepSize = totalPolygonSize / maxAmount;
        double currentPolyCounter = 0;
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::vector<DistPoint> distancePoints;
        distancePoints.resize(totalAmountOfPoints);
        int distIndex = 0;
        std::cout << "Sample points" << std::endl;
        BoundingBox bigBB({-1, -1, -1}, {1, 1, 1});
        for(const auto &poly: polygons){
            const double size = poly.size();
            if(size > stepSize){
                // as long as the step counter is smaller than the end point of the current
                double currentStepCounter = currentPolyCounter;
                for(; currentStepCounter < size + currentPolyCounter;
                      currentStepCounter += stepSize){
                    auto point = poly.getPointOnTriangle(gen);
                    const double distance = dis(gen) * truncationThreshold;
                    point = point + poly.getNormal() * distance;
                    if(distIndex < distancePoints.size()){
                        distancePoints[distIndex] = DistPoint(point[0], point[1], point[2], distance);
                        ++distIndex;
                    }
                }
                currentPolyCounter = fmod(currentStepCounter, stepSize);
            }else{
                // only one step
                currentPolyCounter += size;
                if(currentPolyCounter > stepSize){
                    auto point = poly.getPointOnTriangle(gen);
                    const double distance = dis(gen) * truncationThreshold;
                    point = point + poly.getNormal() * distance;
                    if(distIndex < distancePoints.size()){
                        distancePoints[distIndex] = DistPoint(point[0], point[1], point[2], distance);
                        ++distIndex;
                    }
                    currentPolyCounter = 0;
                }
            }
        }
        std::stringstream str4;
        str4 << "Done with sampling of " << totalAmountOfPoints << " points";
        StopWatchPrinter swp5(str4.str());
        std::uniform_real_distribution<> zeroToOneDis(0.0, 1.0);
        std::normal_distribution<double> normalDis(0.0, 1.0);
        const double currentAmountOfPoints = distIndex;
        for(int i = distIndex; i < totalAmountOfPoints; ++i){
            bool foundAPoint = false;
            while(!foundAPoint){
                const int currentSampledIndex = int(zeroToOneDis(gen) * currentAmountOfPoints);
                const auto &usedPoint = distancePoints[currentSampledIndex];

                dPoint dir;
                double totalLen = 0.0;
                for(int j = 0; j < 3; ++j){
                    dir[j] = normalDis(gen);
                    totalLen += dir[j] * dir[j];
                }
                dir /= sqrt(totalLen);

                dir *= zeroToOneDis(gen) * truncationThreshold * 2.0;
                const dPoint finalPoint = usedPoint + dir;
                if(bigBB.isPointIn(finalPoint)){
                    distancePoints[i] = DistPoint(finalPoint[0], finalPoint[1], finalPoint[2], 0.0);
                    foundAPoint = true;
                }
            }
        }
        swp5.finish();

        Octree octree = Space::createOctree(polygons, minimumOctreeVoxelSize);
        std::vector<QueueElement> openList;
        std::vector<const Octree *> neighbours;
        const int totalNumberOfLeafs = octree.resetIds();
        std::vector<bool> visitedNodes(totalNumberOfLeafs);
        openList.reserve(totalNumberOfLeafs);
        std::stringstream str2;
        str2 << "Real calc for cam index: " << camPoseIndex;
        StopWatchPrinter swp2(str2.str());
        for(auto &point: distancePoints){
            if(not bigBB.isPointIn(point)){
                for(int i = 0; i < 3; ++i){
                    if(point[i] < -1.0){
                        point[i] = -1.0;
                    }else if(point[i] > 1.0){
                        point[i] = 0.999999;
                    }
                }
            }
            const Polygon *closestPoly = nullptr;
            const double distance = Space::interCalcDistExactlyForPoint(point, octree, openList, neighbours,
                                                                        visitedNodes, closestPoly,
                                                                        maxDistanceToMinPosDist);
            if(closestPoly != nullptr){
                point.setPolygon(closestPoly);
            }else{
                point.setPolygon(nullptr);
            }
            point.setDist(std::min(truncationThreshold, std::max(-truncationThreshold, distance)));
        }
        swp2.finish();

        StopWatchPrinter swp3("Putting into the space");
        BoundingBox bb({0, 0, 0}, {1, 1, 1});
        Array3D<std::list<DistPoint> > space({spaceResolution, spaceResolution, spaceResolution});
        const double voxelSize = 1.0 / spaceResolution;
        for(const auto &point: distancePoints){
            const auto movedPoint = (point * 0.5) + 0.5; // from 0 to 1
            if(bb.isPointIn(movedPoint)){
                const auto index = iPoint(pointFloor(movedPoint * spaceResolution));
                const auto lowerPoint = dPoint(index) / spaceResolution;
                const auto shiftedPoint = (movedPoint - lowerPoint) / voxelSize;
                space(index).emplace_back(shiftedPoint[0], shiftedPoint[1], shiftedPoint[2],
                                          point.getDist(), point.getPolygon());
            }else{
                std::cout << movedPoint << std::endl;
            }
        }
        swp3.finish();
        std::stringstream str;
        str << "Writing " << camPoseIndex << ".hdf5";
        StopWatchPrinter swp1(str.str());
        hid_t fileId = Hdf5Writer::openFile(outputFolder + "/" + "result_" + nr.str() + ".hdf5");
        Hdf5Writer::writeArrayToFile(fileId, space);
        H5Fclose(fileId);

        swp1.finish();

        //std::ofstream file;
        //file.open(outputFolder + "/" + "result_" + nr.str() + ".ply");

        //file << "ply\nformat ascii 1.0" << "\n";
        //file << "element vertex " << totalAmountOfPoints << "\n";
        //file << "property float32 x\n";
        //file << "property float32 y\n";
        //file << "property float32 z\n";
        //file << "property uchar red\n";
        //file << "property uchar green\n";
        //file << "property uchar blue\n";
        //file << "end_header\n";
        //for(const auto& point: distancePoints){
        //    const auto color = rgb((point.getDist()+truncationThreshold) / (2*truncationThreshold));
        //    file << point[0] << " " << point[1] << " " << point[2] << " " << color[0] << " " << color[1] << " " << color[2] << "\n";
        //}
        //file.close();
    }
}

void convertCamPoses(int start, int end, const std::vector<CamPose> &camPoses, PolygonReader& polygonReader,
                     std::string outputFolder, double scaling, double projectionNearClipping,
                     double projectionFarClipping, double nearClipping, double farClipping,
                     double surfacelessPolygonsThreshold, unsigned int spaceResolution, double minimumOctreeVoxelSize,
                     double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads, bool useExactAlgorithm,
                     int approximationAccuracy, double truncationThreshold){
    for(int camPoseIndex = start; camPoseIndex < end; camPoseIndex++){
        const CamPose &camPose = camPoses[camPoseIndex];
        Polygons polygons(
                readInAndPreparePolygons(camPose, polygonReader, scaling, projectionNearClipping, projectionFarClipping,
                                         nearClipping, farClipping, surfacelessPolygonsThreshold));


        Space space({spaceResolution, spaceResolution, spaceResolution}, d_negOnes, d_ones * 2);
        if(useExactAlgorithm)
            space.calcDistsExactly(polygons, minimumOctreeVoxelSize, maxDistanceToMinPosDist, threads,
                                   truncationThreshold);
        else
            space.calcDistsApproximately(polygons, maxDistanceToMinPosDist, truncationThreshold, approximationAccuracy);

        space.correctInnerVoxel(boundaryWidth, truncationThreshold);

        const auto size = space.getData().getSize();
        Array3D<float> rotatedAndFlipped(size);
        Array3D<unsigned short> rotatedAndFlippedObjClass(size);
        for(unsigned int i = 0; i < size[0]; ++i){
            for(unsigned int j = 0; j < size[1]; ++j){
                for(unsigned int k = 0; k < size[2]; ++k){
                    rotatedAndFlipped(i, j, size[2] - 1 - k) = space.getData()(j, size[1] - 1 - i, k);
                    rotatedAndFlippedObjClass(i, j, size[2] - 1 - k) = (unsigned short) space.getObjClassData()(j,
                                                                                                                size[1] -
                                                                                                                1 - i,
                                                                                                                k);
                }
            }
        }

        // convert to 16 bit -> to save memory
        Array3D<unsigned short> compressedArray(size);
        for(unsigned int i = 0; i < size[0]; ++i){
            for(unsigned int j = 0; j < size[1]; ++j){
                for(unsigned int k = 0; k < size[2]; ++k){
                    // clipping some of the values are below the negative trunc value
                    const auto newVal =
                            (std::max(-(float) truncationThreshold, rotatedAndFlipped(i, j, k)) + truncationThreshold) /
                            (2 * truncationThreshold);
                    compressedArray(i, j, k) = (unsigned short) (newVal * 65535.F);
                }
            }
        }
        const std::string outputFilePath = outputFolder + "/output_" + Utility::toString(camPoseIndex) + ".hdf5";
        Hdf5Writer::writeArrayToFile(outputFilePath, compressedArray, rotatedAndFlippedObjClass);
        printMsg("Save output file!");
    }
}


#endif //SDFGEN_OLDCONVERTCAMERAPOSESFUNCTIONS_H
