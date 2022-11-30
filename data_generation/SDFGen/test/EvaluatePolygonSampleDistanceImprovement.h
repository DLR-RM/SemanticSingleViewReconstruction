//
// Created by max on 20.09.21.
//

#ifndef SDFGEN_EVALUATEPOLYGONSAMPLEDISTANCEIMPROVEMENT_H
#define SDFGEN_EVALUATEPOLYGONSAMPLEDISTANCEIMPROVEMENT_H

void test(){
    dPoint point1{-1.0, 0.0, -1.0};
    dPoint point2{1.0, 0.0, 1.0};
    dPoint point3{-1.0, 1.0, -1.0};
    std::vector<Point3D> polyPoints;
    polyPoints.emplace_back(point1, 0);
    polyPoints.emplace_back(point2, 1);
    polyPoints.emplace_back(point3, 2);
    Polygon poly(polyPoints, 1);

    std::vector<DistPoint> base_points;
    int basePointAmount = 15;
    for(unsigned int i = 0 ; i < basePointAmount; ++i){
        const dPoint p = (point2 - point1) * (i / float(basePointAmount)) + point1;
        base_points.emplace_back(p[0], p[1], p[2], 0, &poly);
    }
    poly.calcNormal();


    //std::vector<DistPoint> distPoints;
    //const dPoint a = point2 - point1;
    //const dPoint b = point3 - point1;
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uniform0To1(-1, 1.0);
    //for(unsigned int i = 0; i < 5; ++i){
    //    while(true){
    //        double fac = uniform0To1(gen);
    //        double fac1 = uniform0To1(gen);
    //        if(fac + fac1 < 1.0){
    //            const dPoint newPoint = a * fac + b * fac1 + point1;
    //            distPoints.emplace_back(DistPoint(newPoint[0], newPoint[1], newPoint[2], DBL_MAX,  &poly));
    //            break;
    //        }
    //    }
    //}
    for(auto &point : base_points){
        Mapping::mapPoint(point);
    }

    const int pointMaxLevel = 2;


    std::vector<dPoint> distPoints;
    for(unsigned int i = 0; i < 10; ++i){
        distPoints.emplace_back(uniform0To1(gen), 0, uniform0To1(gen)); //uniform0To1(gen));
    }

    for(auto &point : distPoints){
        Mapping::mapPoint(point);
    }

    std::vector<DistPoint> finalPoints;
    std::vector<DistPoint> closerFinalPoints;
    for(const auto& point: distPoints){
        PointDist pointDist(DBL_MAX, nullptr);
        for(const auto& p: base_points){
            const double dist = (p-point).lengthSquared();
            if(pointDist.first > dist){
                pointDist.first = dist;
                pointDist.second = &p;
            }
        }
        DistPoint p((*pointDist.second)[0], (*pointDist.second)[1], (*pointDist.second)[2], sqrt(pointDist.first), pointDist.second->getPolygon());
        finalPoints.emplace_back(p);
        dPoint recordStartPoint(p);
        DistPoint startPoint(p);
        double newSmallerDist = p.getPolygon()->sampleCloserPoints(startPoint, point, gen);
        startPoint.setDist(newSmallerDist);
        Mapping::mapPoint(startPoint);
        closerFinalPoints.emplace_back(startPoint);
        std::cout << "########################" << std::endl;
        std:: cout << "start: " << recordStartPoint << ", end: " << startPoint << std::endl;
    }

    std::vector<DistPoint> distPointsAsDistPoints;
    for(auto& point: distPoints){
        distPointsAsDistPoints.emplace_back(point[0], point[1], point[2], 0, &poly);
    }

    hid_t fileId = Hdf5Writer::openFile("test_mini_result.hdf5");
    Hdf5Writer::writeVectorToFileId(fileId, "randomPoints", distPointsAsDistPoints);
    Hdf5Writer::writeVectorToFileId(fileId, "points", finalPoints);
    Hdf5Writer::writeVectorToFileId(fileId, "opt_points", closerFinalPoints);
    Hdf5Writer::writeVectorToFileId(fileId, "base_points", base_points);
    H5Fclose(fileId);
    exit(0);


}

#endif //SDFGEN_EVALUATEPOLYGONSAMPLEDISTANCEIMPROVEMENT_H
