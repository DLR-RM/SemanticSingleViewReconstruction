//
// Created by max on 29.09.21.
//

#ifndef SDFGEN_FRONT3DREADER_H
#define SDFGEN_FRONT3DREADER_H


#include <string>
#include "ObjReader.h"
#include "../util/ObjWriter.h"
#include "PolygonReader.h"

class Front3DReader: public PolygonReader {
public:
    Front3DReader(std::string  positionTxtFilePath, std::string  wallObjectFilePath);

    static dTransform extractTransformationFromLine(const std::string& line);

    void read() override;

    void addInfoToHdf5Container(int64_t fileId) override;


private:
    const std::string m_positionTxtFilePath;
    const std::string m_wallObjectsFilePath;
};


#endif //SDFGEN_FRONT3DREADER_H
