//
// Created by max on 20.10.21.
//

#ifndef BLOCKER_BLOCKER_H
#define BLOCKER_BLOCKER_H

#include <utility>
#include <vector>
#include <array>
#include <cfloat>
#include "Array3D.h"
#include "Point.h"

class BoundingBox{
public:
    BoundingBox(dPoint min, dPoint max): m_min(std::move(min)), m_max(std::move(max)){}

    bool isInside(const dPoint checkPoint) const {
        return m_min[0] <= checkPoint[0] && checkPoint[0] < m_max[0] &&
               m_min[1] <= checkPoint[1] && checkPoint[1] < m_max[1] &&
               m_min[2] <= checkPoint[2] && checkPoint[2] < m_max[2];
    }
private:
    const dPoint m_min;
    const dPoint m_max;
};

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class Blocker{
public:
    Blocker(std::vector<dPoint>&& points, std::vector<double>&& distances, std::vector<unsigned char>&& classes,
            const std::string& blockSelectionMode): m_points(std::move(points)), m_classes(std::move(classes)),
            m_distances(std::move(distances)), m_truncationThreshold(m_orgThreshold / (2.0 / m_resolution)),
            blockSelectionMode(blockSelectionMode){
        m_truncationBoundarySelectionValue = m_truncationThreshold * 1.0;
        m_minPointAmountPerBlock = int(0.25 * m_pointAmount);
    }

    Array3D<std::vector<std::vector<Point3D> >> splitDataset();


private:
    constexpr static const int m_resolution = 16;
    constexpr static const int m_pointAmount = 2048;
    constexpr static const int m_batchSize = 8;
    constexpr static const double m_boundaryScaleFactorHalf = 1.1;
    constexpr static const double m_boundaryScaleFactor = 1.2;
    constexpr static const double m_orgThreshold = 0.1;

    const double m_truncationThreshold;
    int m_minPointAmountPerBlock;
    double m_truncationBoundarySelectionValue;
    std::vector<dPoint> m_points;
    std::vector<double> m_distances;
    std::vector<unsigned char> m_classes;

    const std::string& blockSelectionMode;
};


#endif //BLOCKER_BLOCKER_H
