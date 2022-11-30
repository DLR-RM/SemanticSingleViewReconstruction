//
// Created by max on 09.09.20.
//

#ifndef SDFGEN_BEAM_H
#define SDFGEN_BEAM_H


#include "math/Point.h"

class Beam{
public:
    Beam(dPoint startPoint, const dPoint& endPoint): m_start(std::move(startPoint)), m_dir(endPoint - startPoint){
        m_length = m_dir.length();
        if(m_length > 0){
            m_dir /= m_length;
        }
    }

    dPoint getBase() const {
        return m_start;
    }

    dPoint getDir() const {
        return m_dir;
    }

    double getLength() const {
        return m_length;
    }

private:
    dPoint m_start;
    dPoint m_dir;
    double m_length;
};


#endif //SDFGEN_BEAM_H
