//
// Created by max on 21.12.21.
//

#ifndef SDFGEN_NORMALCHECKERRAYCASTER_H
#define SDFGEN_NORMALCHECKERRAYCASTER_H


#include "../geom/Polygon.h"
#include "../container/Octree.h"
#include "../container/Space.h"
#include "StopWatchPrinter.h"

class NormalCheckerRayCaster{
public:
    NormalCheckerRayCaster(const Polygons& polygons): m_polygons(polygons) {};

    /**
     * Calculates the amount of normals which point in the wrong direction
     *
     * @param sideResolution The side resolution of the amount of rays which should be checked, there are always sideResolution**2 checks performed
     * @return the amount of normals in percent which have a positive normal component
     */
    double calcAmountOfBrokenNormals(int sideResolution);

    void calcNormalImageForScene(const int sideResolution, const std::string& filePath);

private:
    const Polygons& m_polygons;
};


#endif //SDFGEN_NORMALCHECKERRAYCASTER_H
