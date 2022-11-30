//
// Created by max on 23.09.21.
//

#ifndef LOSSCALCULATOR_LOSSVALUETYPE_H
#define LOSSCALCULATOR_LOSSVALUETYPE_H

#include <sstream>

constexpr const static int MAX_SEE_AROUND_EDGES_DIST = 150;

class LossValueType {
public:
    // these values should start from 0 upwards
    constexpr static unsigned char wallValue() { return 0; }
    constexpr static unsigned char trueWallValue() { return 1; }
    constexpr static unsigned char freeValue() { return 2; }
    constexpr static unsigned char wallValueNotVisible() { return 3; }
    constexpr static unsigned char trueWallValueNotVisible() { return 4; }
    constexpr static unsigned char notReachable() { return 5; }

    // is determined by the other values, one higher than the highest other value
    constexpr static unsigned char freeNotVisibleStart() { return 6; }
    constexpr static unsigned char freeNotVisibleEnd() {
        if(int(freeNotVisibleStart()) + MAX_SEE_AROUND_EDGES_DIST < 256){
            return freeNotVisibleStart() + MAX_SEE_AROUND_EDGES_DIST;
        }else{
            return 255;
        }
    }

	static unsigned char getClassValue(const unsigned char classId, const int amountOfClasses){
		return freeNotVisibleEnd() + classId + amountOfClasses;
	}

	static unsigned char getClassValueNotVisible(const unsigned char classId, const int amountOfClasses){
		return freeNotVisibleEnd() + classId + amountOfClasses * 3;
	}

	static unsigned char getClassValueTrueWall(const unsigned char classId){
		return freeNotVisibleEnd() + classId;
	}

	static unsigned char getClassValueNotVisibleTrueWall(const unsigned char classId, const int amountOfClasses){
		return freeNotVisibleEnd() + classId + amountOfClasses * 2;
	}

    /**
     * Returns the unsigned char value used for this free value
     * @param distanceValue distance value to the closest free surface
     * @return unsigned char as value for this particular value
     */
    static unsigned char freeNotVisible(const unsigned char distanceValue) {
        double fac = double(distanceValue) / double(MAX_SEE_AROUND_EDGES_DIST);
        if(fac > 1.0){
            fac = 1.0;
        }
        const unsigned char offset = LossValueType::freeNotVisibleStart(); // this is determined by the other type values
        const unsigned char maxValue = LossValueType::freeNotVisibleEnd();
        return (unsigned char) (fac * (maxValue - offset)) + offset;
    }

    static std::string mapToString(const unsigned char value){
        if(value == freeValue()){
            return "free";
        }else if(value == wallValue()){
            return "wall";
        }else if(value == trueWallValue()){
            return "true wall";
        }else if(value == wallValueNotVisible()){
            return "wall value not visible";
        }else if(value == trueWallValueNotVisible()){
            return "true wall value not visible";
        }else if(value == notReachable()){
            return "not reachable";
        }else if(value >= freeNotVisibleStart() && value <= freeNotVisibleEnd()){
            const float factor = float(value - freeNotVisibleStart()) / (freeNotVisibleEnd() - freeNotVisibleStart());
            std::stringstream str;
            str << "free not visible with fac: " << factor;
            return str.str();
        }else{
            std::stringstream str;
            str << "unknown: " << value;
            return str.str();
        }
    }
};




#endif //LOSSCALCULATOR_LOSSVALUETYPE_H
