//
// Created by max on 30.09.21.
//

#ifndef SDFGEN_POLYGONREADER_H
#define SDFGEN_POLYGONREADER_H

class PolygonReader{
public:

    virtual void read() = 0;

    Polygons& getPolygon(){ return m_polygons; }

    virtual void addInfoToHdf5Container(int64_t fileId) = 0;

protected:

    Polygons m_polygons;

};

#endif //SDFGEN_POLYGONREADER_H
