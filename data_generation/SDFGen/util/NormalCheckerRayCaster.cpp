//
// Created by max on 21.12.21.
//

#include "NormalCheckerRayCaster.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double NormalCheckerRayCaster::calcAmountOfBrokenNormals(const int sideResolution){
    StopWatchPrinter swp("Calc ray cast");
    // first split the polygons into an octree to make the raycasting easier
    const int octreeRes = 128;
    Octree octree = Space::createOctree(m_polygons, 2.0 / octreeRes);

    const double scale = 2.0 / sideResolution;
    const Polygon surfaceTestPolygon({{d_zeros, 0},
                                      {d_zeros, 1},
                                      {d_zeros, 2}}, -1);
    int normalWrongCounter = 0;
    int usedCounter = 0;
    std::vector<std::vector<bool> > polyUsed;
    polyUsed.resize(sideResolution);
    for(int i = 0; i < sideResolution; ++i){
        polyUsed[i].resize(sideResolution);
        std::fill(polyUsed[i].begin(), polyUsed[i].end(), false);
    }
    for(int i = 0; i < sideResolution; ++i){
        for(int j = 0; j < sideResolution; ++j){
            dPoint startPoint(i * scale - 1.0, j * scale - 1.0, -1);
            dPoint endPoint(i * scale - 1.0, j * scale - 1.0, 1);
            startPoint.clipToMaxAbsValue(1-1e-7);
            endPoint.clipToMaxAbsValue(1-1e-7);
            const Polygon* polyPtr = octree.rayIntersectPolygon(startPoint, endPoint, surfaceTestPolygon);
            if(polyPtr != nullptr){
                Beam beam(startPoint, endPoint);
                // if the z coordinate is positive it points away from the camera and creates a whole in the mesh
                if(polyPtr->getNormal()[2] > 0.0){
                    ++normalWrongCounter;
                    polyUsed[sideResolution - j - 1][i] = true;
                }
                ++usedCounter;
            }
        }
    }
    // remove fireflies, which do not have any connection to other pixels
    for(int i = 1; i < sideResolution - 1; ++i){
        for(int j = 1; j < sideResolution - 1; ++j){
            if(polyUsed[i][j]){
                if(!polyUsed[i-1][j] && !polyUsed[i+1][j] && !polyUsed[i][j-1] && !polyUsed[i][j+1]){
                    polyUsed[i][j] = false;
                    --normalWrongCounter;
                }
            }
        }
    }
    swp.finish();
    const double relativeBroken = normalWrongCounter / (double) usedCounter * 100.0;
    printMsg("Amount of wrong normals: " << normalWrongCounter << ", relative: " << relativeBroken << "%");
    return relativeBroken;
}

void NormalCheckerRayCaster::calcNormalImageForScene(const int sideResolution, const std::string &filePath){
    StopWatchPrinter swp("Calc ray cast");
    // first split the polygons into an octree to make the raycasting easier
    const int octreeRes = 128;
    Octree octree = Space::createOctree(m_polygons, 2.0 / octreeRes);

    const double scale = 2.0 / sideResolution;
    const Polygon surfaceTestPolygon({{d_zeros, 0},
                                      {d_zeros, 1},
                                      {d_zeros, 2}}, -1);
    std::vector<std::vector<dPoint> > polyDist;
    polyDist.resize(sideResolution);
    std::vector<std::vector<bool> > polyUsed;
    polyUsed.resize(sideResolution);
    std::vector<std::vector<bool> > polyBroken;
    polyBroken.resize(sideResolution);
    for(int i = 0; i < sideResolution; ++i){
        polyDist[i].resize(sideResolution);
        polyUsed[i].resize(sideResolution);
        polyBroken[i].resize(sideResolution);
        std::fill(polyDist[i].begin(), polyDist[i].end(), d_zeros);
        std::fill(polyUsed[i].begin(), polyUsed[i].end(), false);
        std::fill(polyBroken[i].begin(), polyBroken[i].end(), false);
    }
    for(int i = 0; i < sideResolution; ++i){
        for(int j = 0; j < sideResolution; ++j){
            dPoint startPoint(i * scale - 1.0, j * scale - 1.0, -1);
            dPoint endPoint(i * scale - 1.0, j * scale - 1.0, 1);
            startPoint.clipToMaxAbsValue(1 - 1e-7);
            endPoint.clipToMaxAbsValue(1 - 1e-7);
            const Polygon *polyPtr = octree.rayIntersectPolygon(startPoint, endPoint, surfaceTestPolygon);
            if(polyPtr != nullptr){
                Beam beam(startPoint, endPoint);
                for(int k = 0; k < 3; ++k){
                    polyDist[sideResolution - j - 1][i][k] = polyPtr->getNormal()[k];
                }
                polyUsed[sideResolution - j - 1][i] = true;
                if(polyPtr->getNormal()[2] > 0.0){
                    polyBroken[sideResolution - j - 1][i] = true;
                }

            }
        }
    }
    // remove fireflies, which do not have any connection to other pixels
    for(int i = 1; i < sideResolution - 1; ++i){
        for(int j = 1; j < sideResolution - 1; ++j){
            if(polyBroken[i][j]){
                if(!polyBroken[i-1][j] && !polyBroken[i+1][j] && !polyBroken[i][j-1] && !polyBroken[i][j+1]){
                    polyBroken[i][j] = false;
                }
            }
        }
    }

    swp.finish();
    std::vector<uint8_t> imgData;
    std::vector<uint8_t> imgDataWrong;
    imgData.resize(sideResolution * sideResolution * 3);
    imgDataWrong.resize(sideResolution * sideResolution * 3);
    for(int i = 0; i < sideResolution; ++i){
        for(int j = 0; j < sideResolution; ++j){
            if(polyUsed[i][j] && !polyBroken[i][j]){
                for(int k = 0; k < 3; ++k){
                    const int nr = i * sideResolution * 3 + j * 3 + k;
                    imgData[nr] = (uint8_t) ((polyDist[i][j][k] * 0.5 + 0.5) * 255.0);
                    imgDataWrong[nr] = polyBroken[i][j] ? 255 : 0;
                }
            }else{
                const int nr1 = i * sideResolution * 3 + j * 3;
                imgData[nr1] = polyBroken[i][j] ? 255 : 0;
                for(int k = 0; k < 3; ++k){
                    const int nr = i * sideResolution * 3 + j * 3 + k;
                    imgDataWrong[nr] = polyBroken[i][j] ? 255 : 0;
                }
            }
        }
    }

    stbi_write_jpg(filePath.c_str(), sideResolution, sideResolution, 3, &imgData[0], 100);
    std::cout << "Image written to " << filePath << std::endl;

    std::string brokenFilePath(filePath);
    brokenFilePath.replace(brokenFilePath.rfind(".jpg"), 4, std::string("_broken.jpg"));
    stbi_write_jpg(brokenFilePath.c_str(), sideResolution, sideResolution, 3, &imgDataWrong[0], 100);
    std::cout << "Just correct image written to " << brokenFilePath << std::endl;
}
