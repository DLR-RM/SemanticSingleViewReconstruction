//
// Created by max on 04.02.22.
//

#ifndef LOSSCALCULATOR_CLASSPOINT_H
#define LOSSCALCULATOR_CLASSPOINT_H

#include "Point.h"

class ClassPoint: public Point<double> {
public:
    ClassPoint(double x, double y, double z, unsigned char classId): Point<double>(x, y, z), m_classId(classId){};

    void setClassId(unsigned char classId){m_classId = classId; }

    unsigned char getClassId() const { return m_classId; }

private:

    unsigned char m_classId;

};

std::ostream& operator<<(std::ostream& os, const ClassPoint& point);

#endif //LOSSCALCULATOR_CLASSPOINT_H
