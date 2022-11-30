//
// Created by max on 30.04.20.
//

#include "ObjWriter.h"

void ObjWriter::write(const Polygons& polygons) {
    std::ofstream file;
    file.open(m_filePath);
    unsigned int total_counter = 1;
    for(const auto& polygon : polygons){
        const auto& points = polygon.getPoints();
        for(const auto point : points) {
            file << "v " << point[0] << " " << point[1] << " " << point[2] << "\n";
        }
        file << "f " << total_counter << " " << total_counter + 1 << " " << total_counter + 2 << "\n";
        total_counter += 3;
    }
    file.close();
}

void ObjWriter::write(const Polygons& polygons, const std::vector<bool>& usePoly){
    if(usePoly.size() != polygons.size()){
        printError("The size in the writting does not match!");
        return;
    }
    std::ofstream file;
    file.open(m_filePath);
    unsigned int total_counter = 1;
    for(unsigned int i = 0; i < polygons.size(); ++i){
        if(usePoly[i]){
            const auto &points = polygons[i].getPoints();
            for(const auto point : points){
                file << "v " << point[0] << " " << point[1] << " " << point[2] << "\n";
            }
            file << "f " << total_counter << " " << total_counter + 1 << " " << total_counter + 2 << "\n";
            total_counter += 3;
        }
    }
    file.close();
}

