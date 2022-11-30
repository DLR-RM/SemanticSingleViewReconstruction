//
// Created by max on 30.04.20.
//

#ifndef SDFGEN_OBJWRITER_H
#define SDFGEN_OBJWRITER_H


#include <string>
#include "../geom/Polygon.h"

class ObjWriter {
public:
    ObjWriter(std::string filePath): m_filePath(std::move(filePath)) {};

    void write(const Polygons& polygons);

    void write(const Polygons& polygons, const std::vector<bool>& usePoly);

    template<class T>
    void write(const std::vector<T>& points, int maxPoints = 0);

private:
    std::string m_filePath;
};

template<class T>
void ObjWriter::write(const std::vector<T>& points, int maxPoints){
    std::ofstream file;
    file.open(m_filePath);
    if(maxPoints == 0){
        maxPoints = points.size();
    }
    for(unsigned int i = 0; i < points.size() && i < maxPoints; ++i){
        const auto& point = points[i];
        file << "v " << point[0] << " " << point[1] << " " << point[2] << "\n";
    }
    file.close();
}


#endif //SDFGEN_OBJWRITER_H
