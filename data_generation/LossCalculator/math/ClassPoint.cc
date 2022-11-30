//
// Created by Maximilian Denninger on 2022-02-04.
//

#include "ClassPoint.h"

std::ostream& operator<<(std::ostream& os, const ClassPoint& point){
	const double p = point[0];
	os << "[" << point[0] << ", " << point[1] << ", " << point[2] << " (" << (int) point.getClassId() << ")]";
	return os;
}
