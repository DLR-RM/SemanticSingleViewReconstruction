//
// Created by Maximilian Denninger on 09.08.18.
//

#ifndef SDFGEN_OBJREADER_H
#define SDFGEN_OBJREADER_H

#include <string>
#include <fstream>
#include "../util/Utility.h"
#include "../geom/math/Point.h"
#include <vector>
#include "../geom/Polygon.h"
#include "../geom/BoundingBox.h"

class ObjReader {
public:
	ObjReader() = default;

    void read(const std::string &filePath);

    void readWithObjNamesAsClassNrs(const std::string& filePath);

    template<class T>
    void read(const std::string &filePath, const T& transform, const int &objectClass);

	Polygons& getPolygon(){ return m_polygons; }

	BoundingBox& getBoundingBox() { return m_box; }

	std::vector<Point3D>& getPoints() { return m_points;}

    static void removeStartAndTrailingWhiteSpaces(std::string& line);

private:
	std::vector<Point3D> m_points;

	Polygons m_polygons;

	BoundingBox m_box;

    std::vector<double> default_transformation = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

	bool startsWith(const std::string& line, const std::string& start);

    void extractFaceInformation(std::string& line, const int objectClass);

    template<class T>
    void extractVertexInformation(std::string& line, int& counter, const T& transform){
        line = line.substr(2, line.length() - 2);
        std::stringstream ss;
        ss << line;
        double x, y, z;
        ss >> x >> y >> z;

        Point3D p(x, y, z, counter);
        p.transformPoint(transform);
        ++counter;
        m_points.emplace_back(std::move(p));
    }

    template<class T>
    static double calcDeterminate(const T& transform);
};

template<class T>
double ObjReader::calcDeterminate(const T& t){
    // https://en.wikipedia.org/wiki/Determinant
    return t[0] * t[5] * t[10] +  // a * e * i +
           t[4] * t[9] * t[2] +   // b * f * g +
           t[8] * t[1] * t[6] -   // c * d * h -
           t[8] * t[5] * t[2] -   // c * e * g -
           t[4] * t[1] * t[10] -  // b * d * i -
           t[0] * t[9] * t[6];    // a * f * h
}

template<class T>
void ObjReader::read(const std::string &filePath, const T& transform, const int &objectClass) {
	std::ifstream stream(filePath);
	if(stream.is_open()){
		std::string line;
		int counter = 1;
		while(std::getline(stream, line)){
			std::string oldLine = line;
			removeStartAndTrailingWhiteSpaces(line);
			if(startsWith(line, "v ")){
                extractVertexInformation(line, counter, transform);
			}else if(startsWith(line, "f ")){
                extractFaceInformation(line, objectClass);
			}
		}
        // if the determinant of the transform is negative it contains a reflection
        if(calcDeterminate(transform) < 0.0){
            // flip all polygons
            for(auto& poly: m_polygons){
                poly.flipPointOrder();
                poly.calcNormal();
            }
        }
	}else{
		printError("File \"" << filePath << "\" could not be read");
	}
}


#endif //SDFGEN_OBJREADER_H
