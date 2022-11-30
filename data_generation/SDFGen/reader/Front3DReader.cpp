//
// Created by max on 29.09.21.
//

#include "Front3DReader.h"
#include "../util/Hdf5Writer.h"

#include <utility>

Front3DReader::Front3DReader(std::string positionTxtFilePath, std::string wallObjectFilePath):
        m_positionTxtFilePath(std::move(positionTxtFilePath)), m_wallObjectsFilePath(std::move(wallObjectFilePath)){
}

dTransform Front3DReader::extractTransformationFromLine(const std::string &line){
    std::stringstream ss;
    ss << line;
    dTransform output;
    ss >> output;
    return output;
}

void Front3DReader::read(){
    m_polygons.clear();
    std::ifstream stream(m_positionTxtFilePath);
    if(stream.is_open()){
        std::string line;
        int counter = 1;
        while(std::getline(stream, line)){
            ObjReader::removeStartAndTrailingWhiteSpaces(line);
            if(line.length() > 0){
                dTransform currentTransformation = extractTransformationFromLine(line);
                currentTransformation.transpose();
                auto pos = line.find('/');
                if(pos != std::string::npos){
                    const auto pathToObj = line.substr(pos, line.find(' ', pos + 1) - pos);
                    std::stringstream ss;
                    ss << line.substr(line.rfind(' '), line.length());
                    int classNr;
                    ss >> classNr;
                    ObjReader objReader;
                    objReader.read(pathToObj, currentTransformation.internalValues(), classNr);
                    m_polygons.insert(m_polygons.end(), objReader.getPolygon().begin(), objReader.getPolygon().end());
                }
            }
        }

    }else{
        printError("The position file is missing: " << m_positionTxtFilePath);
        exit(1);
    }
    ObjReader objReader;
    objReader.readWithObjNamesAsClassNrs(m_wallObjectsFilePath);
    m_polygons.insert(m_polygons.end(), objReader.getPolygon().begin(), objReader.getPolygon().end());
}

void Front3DReader::addInfoToHdf5Container(int64_t fileId){
    Hdf5Writer::writeStringToFileId(fileId, "used_position_file", m_positionTxtFilePath);
    Hdf5Writer::writeStringToFileId(fileId, "used_wall_object_file", m_wallObjectsFilePath);
    Hdf5Writer::writeStringToFileId(fileId, "dataset", std::string("3D front"));
}
