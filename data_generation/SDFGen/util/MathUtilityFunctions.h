//
// Created by max on 04.10.21.
//

#ifndef SDFGEN_MATHUTILITYFUNCTIONS_H
#define SDFGEN_MATHUTILITYFUNCTIONS_H

#include "CamPose.h"

Polygons readInAndPreparePolygons(const CamPose &camPose, PolygonReader& polygonReader, double scaling,
                                  double projectionNearClipping, double projectionFarClipping, double nearClipping,
                                  double farClipping, double surfacelessPolygonsThreshold){
    static std::mutex readInMutex;
    readInMutex.lock();
    polygonReader.read();
    // move them out and clear them to make sure that this can be done in parallel
    Polygons polygons(std::move(polygonReader.getPolygon()));
    polygonReader.getPolygon() = Polygons();

    if(scaling != 1)
        scalePoints(polygons, scaling);
    printVars(camPose.camPos, camPose.towardsPose);
    dTransform camTrans;
    camTrans.setAsCameraTransTowards(camPose.camPos, camPose.towardsPose, camPose.upPos);

    dTransform projectionTrans;
    projectionTrans.setAsProjectionWith(camPose.xFov, camPose.yFov, projectionNearClipping, projectionFarClipping);

    transformPoints(polygons, camTrans);
    polygons = nearFarPolygonClipping(polygons, nearClipping, farClipping);

    transformPoints(polygons, projectionTrans, true);

    polygons = removePolygonsOutOfFrustum(polygons);
    polygons = frustumClipping(polygons);
    polygons = removeFlatPolygons(polygons, surfacelessPolygonsThreshold);
    readInMutex.unlock();
    return polygons;
}

template<unsigned int amount>
void calculateMinValuesForAnchorPoints(std::array<DistPoint, amount>& minValues, const std::vector<dPoint>& anchorPoints,
                                       const dPoint& point){
    /**
     * Calculates the closest k anchor points to the given point. These will be sorted in minValues.
     * k is here set by maxAmountOfUsedAnchorPoints.
     */
    std::fill(minValues.begin(), minValues.end(), DistPoint({0,0,0}, DBL_MAX));

    for(const dPoint& anchorPoint: anchorPoints){
        const auto dist = (anchorPoint - point).lengthSquared();
        for(int i = 0; i < amount; ++i){
            if(dist < minValues[i].getDist()){
                // move the points by one value
                if(i + 1 < amount){
                    // j >= i equal not necessary as i is overwritten anyway
                    for(int j = amount - 1; j > i; --j){
                        minValues[j] = minValues[j - 1];
                    }
                }
                minValues[i] = DistPoint(anchorPoint, dist);
                break;
            }
        }
    }
}

//input: ratio is between 0 to 1
//output: rgb color
std::array<int, 3> rgb(double ratio){
    //we want to normalize ratio so that it fits in to 6 regions
    //where each region is 256 units long
    int normalized = int(ratio * 256 * 6);

    //find the distance to the start of the closest region
    int x = normalized % 256;

    int red = 0, grn = 0, blu = 0;
    switch(normalized / 256){
        case 0:
            red = 255;
            grn = x;
            blu = 0;
            break;//red
        case 1:
            red = 255 - x;
            grn = 255;
            blu = 0;
            break;//yellow
        case 2:
            red = 0;
            grn = 255;
            blu = x;
            break;//green
        case 3:
            red = 0;
            grn = 255 - x;
            blu = 255;
            break;//cyan
        case 4:
            red = x;
            grn = 0;
            blu = 255;
            break;//blue
        case 5:
            red = 255;
            grn = 0;
            blu = 255 - x;
            break;//magenta
        case 6:
            red = 255;
            grn = 0;
            blu = 0;
            break;
    }

    return {red, grn, blu};
}


#endif //SDFGEN_MATHUTILITYFUNCTIONS_H
