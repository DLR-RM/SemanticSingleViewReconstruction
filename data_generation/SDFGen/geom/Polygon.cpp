//
// Created by Maximilian Denninger on 13.08.18.
//

#include <cfloat>
#include "Polygon.h"
#include "../util/Utility.h"
#include "BoundingBox.h"
#include "PolygonCubeIntersection.h"

Polygon::Polygon(const iPoint &indices, Points &pointsRef) : m_calcNormal(false) {
    for (const auto &ele : indices) {
        m_points.emplace_back(pointsRef[ele - 1]);
    }
}

Polygon::Polygon(const iPoint &indices, Points &pointsRef, const int &objectClass) : m_calcNormal(false) {
    m_object_class = objectClass;
    for (const auto &ele : indices) {
        m_points.emplace_back(pointsRef[ele - 1]);
    }
}

double Polygon::calcDistanceConst(const dPoint &point) const {
    const double planeDist = m_main.getDist(point);
    const double desiredSign = sgn(planeDist);
    const double zeroVal = 0.000;
    for (unsigned int currentEdgeId = 0; currentEdgeId < 3; ++currentEdgeId) {
        const double dist = m_edgePlanes[currentEdgeId].getDist(point);
        if (dist < zeroVal) { // is outside
            const double firstBorder = m_borderPlanes[currentEdgeId][0].getDist(point);
            const double secondBorder = m_borderPlanes[currentEdgeId][1].getDist(point);
            if (firstBorder < zeroVal) {
                // use the point dist to the first point
                return desiredSign * (point - m_borderPlanes[currentEdgeId][0].m_pointOnPlane).length();
            } else if (secondBorder < zeroVal) {
                // use the point dist to second point
                return desiredSign * (point - m_borderPlanes[currentEdgeId][1].m_pointOnPlane).length();
            } else {
                return desiredSign * m_edgeLines[currentEdgeId].getDist(point);
            }
        }
    }
    return planeDist;
}

double Polygon::calcDistance(const dPoint &point) {
    if (m_calcNormal) {
        return calcDistanceConst(point);
    } else {
        calcNormal();
        if (m_calcNormal) {
            return calcDistanceConst(point);
        }
    }
    return -1;
}

void Polygon::calcNormal() {
    if (m_points.size() == 3) {
        const auto vec01 = m_points[1] - m_points[0];
        const auto vec02 = m_points[2] - m_points[0];

        m_main.calcNormal(vec01, vec02);
        m_main.calcNormalDist(m_points[0]);

        // edgeId = 0 -> edge between 0 and 1
        for (unsigned int edgeId = 0; edgeId < 3; ++edgeId) {
            const unsigned int nextId = (edgeId + 1) % 3;
            const unsigned int nextNextId = (edgeId + 2) % 3;
            const auto first = m_points[nextNextId] - m_points[nextId];
            const auto second = m_points[edgeId] - m_points[nextId];
            const double alpha = acos(dot(first, second) / (first.length() * second.length()));
            dPoint normal;
            if (fabs(alpha - M_PI_2) > 1e-5) {
                const double length = cos(alpha) * second.length();
                const auto dir = first.normalize();
                auto newPoint = dir * length + m_points[nextId];
                normal = m_points[edgeId] - newPoint;
            } else { // they are perpendicular
                normal = second;
            }
            m_edgePlanes[edgeId] = Plane(normal, m_points[nextId]);
            m_edgeLines[edgeId] = Line(m_points[nextNextId], m_points[nextId]);
            m_borderPlanes[edgeId][0] = Plane(first, m_points[nextId]);
            m_borderPlanes[edgeId][1] = Plane(first * (-1), m_points[nextNextId]);
        }

        m_calcNormal = true;
    } else {
        printError("Polygons have to be triangles!");
    }
}


void Polygon::flipPointOrder() {
    std::swap(m_points[0], m_points[1]);
}

double Polygon::rayIntersectDist(const Beam& beam) const {
    if(!m_calcNormal){
        printError("The normals have to be calculated before!");
        exit(1);
    }
    const double divisor = dot(beam.getDir(), this->m_main.m_normal);
    if(divisor==0.)
        return -1;
    //Distance between viewpoint and plane in beam direction
    const double dividend = this->m_main.m_dist - dot(beam.getBase(), this->m_main.m_normal);
    const double dist = dividend / divisor;
    if(dist < 0){
        return -1;
    }
    if(dist > beam.getLength()){
        return -1;
    }
    return dist;
}
bool Polygon::rayIntersect(const Beam& beam) const {
    if(!m_calcNormal){
        printError("The normals have to be calculated before!");
        exit(1);
    }
    const double divisor = dot(beam.getDir(), this->m_main.m_normal);
	if(divisor==0.)
		return false;
	//Distance between viewpoint and plane in beam direction
	const double dividend = this->m_main.m_dist - dot(beam.getBase(), this->m_main.m_normal);
	const double dist = dividend / divisor;
	if(dist < 0){
		return false;
	}
	if(dist > beam.getLength()){
	    return false;
	}
	// intersection point between plane and ray
	const dPoint intersPoint = beam.getBase() + beam.getDir() * dist;
	for(const auto& plane : m_edgePlanes){
		if(plane.getDist(intersPoint) < 0.0){
			// point lays outside the area
			return false;
		}
	}
	return true;

}

dPoint Polygon::getPointOnTriangle(std::mt19937 &gen) const {
    const auto first = m_points[0] - m_points[1];
    const auto second = m_points[2] - m_points[1];
    std::uniform_real_distribution<> dis;
    while(true){
        double a = dis(gen);
        double b = dis(gen);
        const auto newPoint = m_points[1] + first * a + second * b;
        if(m_edgePlanes[1].getDist(newPoint) > 0){
            return newPoint;
        }
    }
}

bool Polygon::checkIfIsAbovePolygon(const dPoint &point) const{
    for (unsigned int currentEdgeId = 0; currentEdgeId < 3; ++currentEdgeId){
        const double dist = m_edgePlanes[currentEdgeId].getDist(point);
        if(dist < 0.000){ // is outside
            return false;
        }
    }
    return true;
}

double Polygon::sampleCloserPoints(const dPoint &startPoint, const dPoint &comparePoint, std::mt19937 &gen) const{
    dPoint currentPoint(startPoint);
    dPoint a = m_points[2] - m_points[0];
    dPoint b = m_points[1] - m_points[0];
    a.normalizeThis();
    b.normalizeThis();
    auto calculateDistToComparePoint = [comparePoint](dPoint currentPoint) -> double {
        Mapping::mapPoint(currentPoint);
        return (comparePoint - currentPoint).lengthSquared();
    };
    double minDist = (comparePoint - currentPoint).lengthSquared();
    Mapping::mapPointBack(currentPoint);
    double stepSize = 0.10;
    std::array<std::pair<dPoint, double> , 8> closerPoints;
    int closerPointIndex;
    for(int i = 0; i < 50; ++i){
        closerPointIndex = 0;
        // calculate the gradient
        for(int j = -1; j < 2; j += 2){
            for(int k = -1; k < 2; k += 2){
                dPoint gradient = a * float(j) + b * float(k);
                gradient.normalizeThis();
                const dPoint newPoint = gradient * stepSize + currentPoint;
                if(checkIfIsAbovePolygon(newPoint)){
                    const double newDist = calculateDistToComparePoint(newPoint);
                    if(newDist < minDist){
                        closerPoints[closerPointIndex] = std::pair<dPoint, double>(gradient, minDist - newDist);
                        ++closerPointIndex;
                    }
                }
            }
        }
        for(int j = 0; j < 4; ++j){
            const int dir1 = int(j % 2 == 0);
            const int dir2 = int((j + 1) % 2 == 0);
            const int sign = int(j > 1) * 2 - 1;
            dPoint gradient = a * float(dir1 * sign) + b * float(dir2 * sign);
            gradient.normalizeThis();
            const dPoint newPoint = currentPoint + gradient * stepSize;
            if(checkIfIsAbovePolygon(newPoint)){
                const double newDist = calculateDistToComparePoint(newPoint);
                if(newDist < minDist){
                    closerPoints[closerPointIndex] = std::pair<dPoint, double>(newPoint - currentPoint, minDist - newDist);
                    ++closerPointIndex;
                }
            }
        }
        if(closerPointIndex > 0){
            dPoint gradient = d_zeros;
            if(closerPointIndex > 1){
                double totalDistChange = 0.0;
                for(int j = 0; j < closerPointIndex; ++j){
                    totalDistChange += closerPoints[j].second;
                }
                for(int j = 0; j < closerPointIndex; ++j){
                    const double fac = closerPoints[j].second / totalDistChange;
                    gradient += closerPoints[j].first * fac;
                }
            }else{
                gradient = closerPoints[0].first;
            }
            gradient.normalizeThis();
            dPoint newPoint = gradient * stepSize + currentPoint;
            const double newDist = calculateDistToComparePoint(newPoint);
            if(newDist < minDist){
                minDist = newDist;
                currentPoint = newPoint;
            }else{
                stepSize *= 0.25;
            }
        }else if(stepSize < 1e-10){
            break;
        }else{
            stepSize *= 0.25;
        }
    }
    return sqrt(minDist);
}


Polygons removePolygonsOutOfFrustum(Polygons &polys, const dPoint cubeMax, const dPoint cubeMin){
    Polygons newPolys;
    for (unsigned int i = 0; i < polys.size(); ++i) {
        if (t_c_intersection(polys[i], cubeMax, cubeMin)) {
            newPolys.emplace_back(polys[i].getPoints(), polys[i].getObjectClass());
        }
    }
    return newPolys;
}

Polygons frustumClipping(Polygons &polys) {
    Polygons tmpPolys;
    Polygons* newPolys = &tmpPolys;
    Polygons* oldPolys = &polys;

    for (int j = 0; j < 3; j++) {
        for (int d = -1; d <= 1; d += 2) {
            for (unsigned int i = 0; i < oldPolys->size(); ++i) {
                int needsClipping = 0;
                for (auto &point : (*oldPolys)[i].getPoints()) {
                    if (point[j] * d > 1) {
                        needsClipping++;
                    }
                }

                if (needsClipping > 0) {
                    dPoint normal(0, 0, 0);
                    dPoint pointOnPlane(0, 0, 0);
                    normal[j] = d;
                    pointOnPlane[j] = d;
                    Plane clippingPlane(normal, pointOnPlane);
                    clipAtPlane(*newPolys, (*oldPolys)[i].getPoints(), (*oldPolys)[i].getObjectClass(), clippingPlane, needsClipping);
                } else {
                    newPolys->emplace_back((*oldPolys)[i].getPoints(), (*oldPolys)[i].getObjectClass());

                }
            }

            std::swap(newPolys, oldPolys);
            newPolys->clear();
        }
    }
    return *oldPolys;
}

Polygons frustumClippingOnBB(Polygons &polys, const dPoint& lowerPoint, const dPoint& upperPoint) {
    Polygons tmpPolys;
    Polygons* newPolys = &tmpPolys;
    Polygons* oldPolys = &polys;

    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; ++k) {
            const double current_dim = k == 0? lowerPoint[j] : upperPoint[j];
            const double d = k == 0 ? -1 : 1;
            for (unsigned int i = 0; i < oldPolys->size(); ++i) {
                int needsClipping = 0;
                for (auto &point : (*oldPolys)[i].getPoints()) {
                    if(k == 0){
                        if(point[j] < current_dim){
                            needsClipping++;
                        }
                    }else{
                        if (point[j] > current_dim) {
                            needsClipping++;
                        }
                    }
                }

                if (needsClipping > 0) {
                    dPoint normal(0, 0, 0);
                    dPoint pointOnPlane(0, 0, 0);
                    normal[j] = d;
                    pointOnPlane[j] = current_dim;
                    Plane clippingPlane(normal, pointOnPlane);
                    clipAtPlane(*newPolys, (*oldPolys)[i].getPoints(), (*oldPolys)[i].getObjectClass(), clippingPlane, needsClipping);
                } else {
                    newPolys->emplace_back((*oldPolys)[i].getPoints(), (*oldPolys)[i].getObjectClass());
                }
            }

            std::swap(newPolys, oldPolys);
            newPolys->clear();
        }
    }
    return *oldPolys;
}


Polygons removeFlatPolygons(Polygons &polys, double threshold) {
    Polygons newPolys;

    for (unsigned int i = 0; i < polys.size(); ++i) {
        auto points = polys[i].getPoints();
        Line line(points[0], points[1]);

        if ((points[0] - points[1]).length() > threshold && line.getDist(points[2]) > threshold) {
            newPolys.emplace_back(polys[i].getPoints(), polys[i].getObjectClass());
        }
    }

    printMsg("Removed polygons with all three vertices on a line: " + std::to_string(polys.size() - newPolys.size()));
    return newPolys;
}

Polygons nearFarPolygonClipping(Polygons &polygons, double nearClipping, double farClipping) {
    Polygons newPolys;
    Plane nearClippingPlane({0, 0, -1}, {0, 0, -nearClipping});
    for (unsigned int i = 0; i < polygons.size(); ++i) {
        bool usePoly = false;
        auto &points = polygons[i].getPoints();

        for (const auto &point : points) {
            if (point[2] < -nearClipping && point[2] > -farClipping) {
                usePoly = true;
            }
        }
        auto &first = points[0];
        if (!usePoly) {
            for (int j = 1; j < 3; ++j) {
                // poly stretches through volume
                if (first[2] >= -nearClipping && points[j][2] <= -farClipping) {
                    usePoly = true;
                    break;
                } else if (first[2] <= -farClipping && points[j][2] >= -nearClipping) {
                    usePoly = true;
                    break;
                }
            }
        }
        int needsClipping = 0;
        for (const auto &point : points) {
            if (usePoly && point[2] > -nearClipping) {
                ++needsClipping;
            }
        }
        if (needsClipping > 0) {
            clipAtPlane(newPolys, points, polygons[i].getObjectClass(), nearClippingPlane, needsClipping);
        } else if (usePoly) {
            newPolys.emplace_back(points, polygons[i].getObjectClass());
        }
    }

    return newPolys;
}

void clipAtPlane(Polygons &newPolys, Points &polygonPoints, const int objectClass, const Plane &clippingPlane, int needsClipping) {
    std::array<Line, 3> lines;
    for (unsigned int i = 0; i < 3; ++i) {
        lines[i] = Line(polygonPoints[i], polygonPoints[(i + 1) % 3]);
    }
    if (needsClipping == 1) {
        int notUsed = -1;
        for (unsigned int i = 0; i < 3; ++i) {
            if (!clippingPlane.intersectBy(lines[i])) {
                notUsed = i;
            }
        }

        dPoint firstCut = clippingPlane.intersectionPoint(lines[(notUsed + 1) % 3]);
        dPoint secondCut = clippingPlane.intersectionPoint(lines[(notUsed + 2) % 3]);
        Point3D firstCut3D(firstCut, 0);
        Point3D secondCut3D(secondCut, 0);
        Points p1 = {polygonPoints[notUsed], polygonPoints[(notUsed + 1) % 3], firstCut3D};
        Points p2 = {firstCut3D, secondCut3D, polygonPoints[notUsed]};
        newPolys.emplace_back(p1, objectClass);
        newPolys.emplace_back(p2, objectClass);
    } else if (needsClipping == 2) {
        int used = -1;
        for (unsigned int i = 0; i < 3; ++i) {
            if (!clippingPlane.intersectBy(lines[i])) {
                used = (i + 2) % 3;
                break;
            }
        }

        dPoint firstCut = clippingPlane.intersectionPoint(lines[used]);
        dPoint secondCut = clippingPlane.intersectionPoint(lines[(used + 2) % 3]); // +2 == -1
        Point3D firstCut3D(firstCut, 0);
        Point3D secondCut3D(secondCut, 0);
        Points p1 = {polygonPoints[used], firstCut3D, secondCut3D};
        newPolys.emplace_back(p1, objectClass);
    }
}


