//
// Created by max on 04.10.21.
//

#ifndef SDFGEN_CAMPOSE_H
#define SDFGEN_CAMPOSE_H

struct CamPose{

    dPoint camPos;
    dPoint towardsPose;
    dPoint upPos;
    double xFov;
    double yFov;

};

std::string convert_to_string(const CamPose& camPose){
    std::stringstream str;
    str << "Cam pose: " << camPose.camPos;
    str << ", towards pose: " << camPose.towardsPose;
    str << ", up pose: " << camPose.upPos;
    str << ", xFov: " << camPose.xFov;
    str << ", yFov: " << camPose.yFov;
    return str.str();
}

std::vector<CamPose> readCameraPoses(std::string cameraPositionsFile){
    std::fstream positions_file(cameraPositionsFile, std::istream::in);
    std::vector<CamPose> camPoses;
    if(positions_file.is_open()){
        std::string line;
        while(positions_file.good()){
            std::getline(positions_file, line); // skip this line
            if(line.length() > 1){
                CamPose pose;
                unsigned int eleCamPos = 0;
                std::string elementString;
                std::istringstream stringStreamEle(line);
                std::cout << line << std::endl;
                while(getline(stringStreamEle, elementString, ' ') && eleCamPos < 11){
                    if(elementString.length() <= 1){
                        continue;
                    }
                    const double ele = std::atof(elementString.c_str());
                    if(eleCamPos < 3){
                        pose.camPos[eleCamPos] = ele;
                    }else if(eleCamPos >= 3 && eleCamPos < 6){
                        pose.towardsPose[eleCamPos - 3] = ele;
                    }else if(eleCamPos >= 6 && eleCamPos < 9){
                        pose.upPos[eleCamPos - 6] = ele;
                    }else if(eleCamPos == 9){
                        pose.xFov = ele;
                    }else if(eleCamPos == 10){
                        pose.yFov = ele;
                    }
                    ++eleCamPos;
                }
                camPoses.emplace_back(pose);
            }
        }
    }else{
        printError("The position file could not be opened: " + cameraPositionsFile);
    }
    return camPoses;
}


#endif //SDFGEN_CAMPOSE_H
