//
// Created by Maximilian Denninger on 13.08.18.
//

#ifndef SDFGEN_POLYGON_H
#define SDFGEN_POLYGON_H


#include "math/Point.h"
#include <vector>
#include "Line.h"
#include "Plane.h"
#include "math/Transform.h"
#include <fstream>
#include <list>
#include <random>
#include "BoundingBox.h"
#include "Beam.h"
#include "math/deform.h"

using Points = std::vector<Point3D>;

class Polygon {
public:
	explicit Polygon(Points points): m_object_class(0), m_points(std::move(points)), m_calcNormal(false){};
    explicit Polygon(Points points, const int object_class): m_object_class(object_class), m_points(std::move(points)), m_calcNormal(false){};

    // indices are in .obj format start with 1
    Polygon(const iPoint& indices, std::vector<Point3D>& pointsRef);
	Polygon(const iPoint& indices, std::vector<Point3D>& pointsRef, const int &objectClass);

	Points& getPoints(){
		return m_points;
	}

	dPoint getNormal() const {
	    return m_main.m_normal;
	}

	const Points& getPoints() const{
		return m_points;
	}

	int getObjectClass() const{
	    return m_object_class;
	}

	double calcDistance3(const dPoint& point) const;

	double calcDistanceConst(const dPoint& point) const;

	double calcDistance(const dPoint& point);

	double calcDistance2(const dPoint& point);

	BoundingBox getBB() const {
		BoundingBox box;
		box.addPolygon(*this);
		return box;
	}

	double size() const{
		const auto first = m_points[0] - m_points[1];
		const auto second = m_points[2] - m_points[1];
		return 0.5 * cross(first, second).length();
	}

	dPoint getPointOnTriangle(std::mt19937& gen) const;

	bool rayIntersect(const Beam& beam) const;

    double rayIntersectDist(const Beam& beam) const;

	void calcNormal();

	void flipPointOrder();

    /**
     * Checks if the point is either in the triangle or directly above or below the triangle.
     * @param point Point which should be checked
     * @return True if the triangle is above or below or on the triangle
     */
    bool checkIfIsAbovePolygon(const dPoint& point) const;

    /**
     * Tries to reduce the distance to a given point on a curved surface, the start point is assumed to be already mapped into the other
     * space. The same is true for the compare point. This is done by projecting the start point back and calculating an estimated gradient
     * by calculating the gradient for eight directions relative to two sides of the triangle. The stepsize is reduced dynamically until no
     * improvement can be reached anymore. At the end the closest possible distance is returned.
     *
     * @param startPoint Start point must be on the curved triangle.
     * @param comparePoint Comparing point will be compared in the curved space
     * @param gen a random number generator to sample new numbers
     * @return the new closest distance
     */
    double sampleCloserPoints(const dPoint& startPoint, const dPoint& comparePoint, std::mt19937& gen) const;

private:

	Points m_points;

	bool m_calcNormal;

	Plane m_main;

	int m_object_class = 0;

	std::array<Plane, 3> m_edgePlanes;

	// for the perpendicular borders at the points
	std::array<std::array<Plane, 2>, 3> m_borderPlanes;

	std::array<Line, 3> m_edgeLines;

};

using Polygons = std::vector<Polygon>;


static void writeToDisc(Polygons& polygons, const std::string& filePath){

	std::ofstream output2(filePath, std::ios::out);

	int counter = 1;
//	BoundingBox cube({-2, -2, -2}, {2, 2, 2});
	for(auto& poly : polygons){
//		bool isInside = false;
//		for(auto& point : poly.getPoints()){
//			if(cube.isPointIn(point)){
//				isInside = true;
//				break;
//			}
//		}
////		isInside = true;
//		if(isInside){
		for(auto& point : poly.getPoints()){
			output2 << "v " << (point)[0] << " " << (point)[1] << " " << (point)[2] << "\n";
			point.setIndex(counter);
			counter += 1;
		}
//		}
	}
	int insideCounter = 0;
	for(auto& poly : polygons){
//		bool isInside = false;
//		for(auto& point : poly.getPoints()){
//			if(cube.isPointIn(point)){
//				isInside = true;
//				break;
//			}
//		}
//
////		isInside = true;
//		if(isInside){
		++insideCounter;
		output2 << "f";
		for(auto& point : poly.getPoints()){
			output2 << " " << point.getIndex();
		}
		output2 << "\n";

//		}
	}

	output2.close();
}

static void transformPoints(Polygons& polygons, const dTransform& trans, bool flipPointOrder = false){
	for(auto& poly: polygons){
		for(unsigned int i = 0; i < 3; ++i){
			auto& point = poly.getPoints()[i];
			trans.transform(point);
		}

		if (flipPointOrder) {
			poly.flipPointOrder();
		}
	}
}

static void scalePoints(Polygons& polygons, double scaling){
    for(auto& poly: polygons){
        for(unsigned int i = 0; i < 3; ++i){
            auto& point = poly.getPoints()[i];
            point *= scaling;
        }
    }
}

Polygons removePolygonsOutOfFrustum(Polygons& polys, const dPoint cubeMax = d_ones + 1e-6, const dPoint cubeMin = d_negOnes - 1e-6);
Polygons removeFlatPolygons(Polygons& polys, double threshold);
Polygons nearFarPolygonClipping(Polygons& polygons, double nearClipping, double farClipping);
Polygons frustumClipping(Polygons &polys);
Polygons frustumClippingOnBB(Polygons &polys, const dPoint& lowerPoint, const dPoint& upperPoint);
void clipAtPlane(Polygons &newPolys, Points &polygonPoints, const int objectClass, const Plane &clippingPlane, int needsClipping);

#endif //SDFGEN_POLYGON_H
