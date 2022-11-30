//
// Created by max on 14.09.20.
//

#ifndef SDFGEN_POINTOCTREE_H
#define SDFGEN_POINTOCTREE_H


#include "../geom/math/Point.h"
#include "../geom/Polygon.h"
#include "Octree.h"
#include "../geom/math/DistPoint.h"

// double is the dist, while int is the classNr
using PointDist = std::pair<double, const DistPoint*>;

class PointOctree{
public:

    PointOctree(std::vector<DistPoint>&& points, int maxLevel, dPoint size, dPoint origin, int level=0);
    bool containsPoint(const dPoint &point) const;

    static bool containsPoint(const dPoint &point, const dPoint& origin, const dPoint& size);

    PointOctree& findNodeContainingPoint(const dPoint &point) const;

    void findNeighbors(PointOctree& currentOctree) const;

    void findExternalNeighbors(PointOctree& currentOctree) const;

    double getDistanceToOctreeSide(const dPoint& point) const;

    static double getDistanceToOctreeSideWith(const dPoint& point, const dPoint& origin, const dPoint& size);

    PointDist distance(const dPoint& point) const;

    OctreeState getState() const {return m_state;}

private:


    void distanceToPoints(const dPoint& point, PointDist& pointDist) const;
    std::vector<PointOctree> m_children;
    std::list<const PointOctree*> m_neighbors;
    std::list<const PointOctree*> m_externalNeighbors;

    bool m_checkedForNeighbors;
    bool m_checkedForExternalNeighbors;
    dPoint m_origin;
    dPoint m_size;
    OctreeState m_state;

    std::vector<DistPoint> m_points;

    static const std::vector<dPoint> m_directions;

    static const BoundingBox m_bigBB;

};



#endif //SDFGEN_POINTOCTREE_H
