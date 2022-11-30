//
// Created by max on 11.02.22.
//

#ifndef SDFGEN_REPLICAREADER_H
#define SDFGEN_REPLICAREADER_H

#include <string>
#include "ObjReader.h"
#include "../util/ObjWriter.h"
#include "PolygonReader.h"
#include "../util/Hdf5Writer.h"


class ReplicaReader: public PolygonReader {

public:
    ReplicaReader(std::string  objectFilePath): m_objectFilePath(objectFilePath){}

    void read(){
        m_polygons.clear();
        ObjReader objReader;
        objReader.readWithObjNamesAsClassNrs(m_objectFilePath);
        m_polygons.insert(m_polygons.end(), objReader.getPolygon().begin(), objReader.getPolygon().end());
    }

    void addInfoToHdf5Container(int64_t fileId){
        Hdf5Writer::writeStringToFileId(fileId, "used_object_file_path", m_objectFilePath);
        Hdf5Writer::writeStringToFileId(fileId, "dataset", std::string("replica"));
    }

private:

    const std::string m_objectFilePath;

};

#endif //SDFGEN_REPLICAREADER_H
