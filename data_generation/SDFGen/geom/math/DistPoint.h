//
// Created by max on 03.09.21.
//

#ifndef SDFGEN_DISTPOINT_H
#define SDFGEN_DISTPOINT_H

#include "Point.h"
#include "../Polygon.h"

class DistPoint : public Point<double> {
public:

    DistPoint() : Point(), m_distance(0.0), m_polygon(nullptr){};
    DistPoint(double x, double y, double z, double distance) : Point<double>(x, y, z), m_distance(distance), m_polygon(nullptr){};

    DistPoint(double x, double y, double z, double distance, const Polygon* poly) : Point<double>(x, y, z), m_distance(distance), m_polygon(poly){};

    DistPoint(const Point<double>& point, double distance): Point<double>(point), m_distance(distance), m_polygon(nullptr){};

    double getDist() const { return m_distance; }

    void setDist(double dist){ m_distance = dist; }

    void sqrtDist(){ m_distance = sqrt(m_distance); }

    int getClass() const { return m_polygon != nullptr ? m_polygon->getObjectClass(): -1; }

    const Polygon* getPolygon() const { return m_polygon; }

    void setPolygon(const Polygon* poly) { m_polygon = poly; }

private:
    double m_distance;

    const Polygon* m_polygon;
};


#endif //SDFGEN_DISTPOINT_H
