//
// Created by max on 30.04.20.
//

#ifndef SDFGEN_CUBE_H
#define SDFGEN_CUBE_H

#include <vector>
#include "Polygon.h"

class Cube {
public:

    Cube(dPoint scale, const std::vector<double> &transformation, int cubeClass=-1){
        m_points = {
                {-1, -1, -1, 1},
                {-1, -1, 1, 2},
                {-1, 1, -1, 3},
                {-1, 1, 1, 4},
                {1, -1, -1, 5},
                {1, -1, 1, 6},
                {1, 1, -1, 7},
                {1, 1, 1, 8},
        };
        for(auto& point : m_points){
            for(unsigned int i = 0; i < 3; ++i){
                point[i] *= scale[i] * -0.5;
            }
            point.transformPoint(transformation);
        }

        // front
        m_polygons.emplace_back(iPoint(1, 2, 5), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(5, 2, 6), m_points, cubeClass);
        // right side
        m_polygons.emplace_back(iPoint(6, 8, 7), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(6, 7, 5), m_points, cubeClass);
        // left side
        m_polygons.emplace_back(iPoint(2, 1, 3), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(4, 2, 3), m_points, cubeClass);
        // back
        m_polygons.emplace_back(iPoint(8, 4, 3), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(8, 3, 7), m_points, cubeClass);
        // top
        m_polygons.emplace_back(iPoint(8, 6, 2), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(8, 2, 4), m_points, cubeClass);
        // bottom
        m_polygons.emplace_back(iPoint(1, 7, 3), m_points, cubeClass);
        m_polygons.emplace_back(iPoint(1, 5, 7), m_points, cubeClass);
    }

    Polygons& getPolygons(){ return m_polygons; }

private:

    Polygons  m_polygons;
    std::vector<Point3D> m_points;

};


#endif //SDFGEN_CUBE_H
