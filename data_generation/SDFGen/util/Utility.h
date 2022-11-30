//
// Created by Maximilian Denninger on 09.08.18.
//

#ifndef SDFGEN_UTILITY_H
#define SDFGEN_UTILITY_H

#include <iostream>
#include <math.h>
#include <sstream>

static std::string convertPrettyFct(std::string str){
	auto pos = str.find("unsigned");
	if(pos != str.npos){
		str = str.substr(0, pos + 1) + str.substr(pos + 9, str.length() - pos - 9);
		return convertPrettyFct(str);
	}
	pos = str.find(" &");
	if(pos != str.npos){
		str = str.substr(0, pos) + str.substr(pos + 1, str.length() - pos - 1);
		return convertPrettyFct(str);
	}
	return str;
}

#define printPrettyFunction() \
    convertPrettyFct(__PRETTY_FUNCTION__) << "::" << __LINE__ \

#define printDetailedMsg(msg, prefix) \
    std::cout << prefix << printPrettyFunction() << ": " << msg << std::endl \

#define printMsg(msg) \
    printDetailedMsg(msg, "") \

#define varCore(var) \
    #var ": " << var \

#define printVar(var) \
    printDetailedMsg(varCore(var), "") \

#define printVars(var1, var2) \
    printDetailedMsg(varCore(var1) << ", " varCore(var2), "")


#define printQuote(msg) \
    printMsg("\"" << msg << "\"") \

#define printError(msg) \
    printDetailedMsg(msg, "Error in ") \

#define printLine() \
    std::cout << "In: " << printPrettyFunction() << std::endl; \

#define printDivider() \
    std::cout << "---------------------------------------------" << std::endl; \

namespace Utility{
	template<typename type>
	static void destroy(type*& ptr){
		delete ptr;
		ptr = nullptr;
	}

	static double deg2rad(double deg){
		return deg * M_PI / 180.0;
	}

	static double rad2deg(double rad){
		return rad * 180. / M_PI;
	}

	template<typename T>
	static std::string toString(const T& object){
		std::stringstream ss;
		ss << object;
		return ss.str();
	}
    #define add_to_string(name) str << "\"" #name "\": \"" << name << "\","

    static std::string convert_to_string_as_yaml(int checkRes, double scaling, double projectionNearClipping,
                                      double projectionFarClipping, double nearClipping,
                                      double farClipping, double surfacelessPolygonsThreshold,
                                      double minimumOctreeVoxelSize,
                                      double maxDistanceToMinPosDist, int boundaryWidth, unsigned int threads,
                                      int approximationAccuracy, double truncationThreshold, bool mapPointsInZDirection,
                                      double relativeBrokenNormalsValue){
        std::stringstream str;
        str << "{";
        add_to_string(checkRes);
        add_to_string(scaling);
        add_to_string(projectionNearClipping);
        add_to_string(projectionFarClipping);
        add_to_string(nearClipping);
        add_to_string(farClipping);
        add_to_string(surfacelessPolygonsThreshold);
        add_to_string(minimumOctreeVoxelSize);
        add_to_string(maxDistanceToMinPosDist);
        add_to_string(boundaryWidth);
        add_to_string(threads);
        add_to_string(approximationAccuracy);
        add_to_string(truncationThreshold);
        add_to_string(mapPointsInZDirection);
        add_to_string(relativeBrokenNormalsValue);
        str << "}";
        return str.str();
    }


}


#endif //SDFGEN_UTILITY_H
