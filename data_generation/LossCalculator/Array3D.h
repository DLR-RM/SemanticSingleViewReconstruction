//
// Created by Maximilian Denninger on 2019-06-17.
//

#ifndef LOSSCALCULATOR_ARRAY3D_H
#define LOSSCALCULATOR_ARRAY3D_H

#include <vector>
#include <iostream>
#include <list>
#include <cmath>
#include <array>
#include <cfloat>
#include "LossValueType.h"
#include "util/StopWatch.h"
#include "src/math/Point.h"

class UnusableException : public std::exception {
	virtual const char* what() const throw() {
    	return "This file can not be used at all!";
  	}
};

using Pose2D = std::pair<unsigned int, unsigned int>;

struct Pose3D{
	unsigned int first;
	unsigned int second;
	unsigned int third;
	Pose3D(unsigned int i, unsigned int j, unsigned int k): first(i),second(j), third(k){}

	Pose3D(unsigned int i, unsigned int j,unsigned int dim, unsigned int value){
		if(dim == 0){
			first = value;
			second = i;
			third = j;
		}else if(dim == 1){
			first = i;
			second = value;
			third = j;
		}else{
			first = i;
			second = j;
			third = value;
		}
	}

	Pose3D(const Pose2D& pose,unsigned int dim, unsigned int value){
		if(dim == 0){
			first = value;
			second = pose.first;
			third = pose.second;
		}else if(dim == 1){
			first = pose.first;
			second = value;
			third = pose.second;
		}else{
			first = pose.first;
			second = pose.second;
			third = value;
		}
	}

	Pose3D createAndMove(unsigned int dim, unsigned int value) const {
		auto res = Pose3D(*this);
		if(dim == 0){
			res.first += value;
		}else if(dim == 1){
			res.second += value;
		}else{
			res.third += value;
		}
		return res;
	}

	unsigned int convertBackToInner(const unsigned int size) const {
		return first * size * size + second * size + third;
	}
};

template<typename T>
class Array3D {
public:
	Array3D(){
		m_innerSize = 0;
		m_size = 0;
		m_values.resize(1);
	}

	explicit Array3D(unsigned int size){
		m_size = size;
		m_innerSize = size * size * size;
		m_values.resize(m_innerSize);
	}

	template <typename ArrayType>
	Array3D(ArrayType* array, unsigned int size){
		init<ArrayType>(array, size);
	}

	template <typename ArrayType>
	Array3D(Array3D<ArrayType>& array){
		init<ArrayType>(&array.getInner(0), array.length());
	}

	template <typename ArrayType>
	void init(ArrayType* array, unsigned int size){
		m_values.clear();
		m_size = size;
		m_innerSize = size * size * size;
		m_values.resize(m_innerSize);
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = T(array[i]);
		}
	}

	void scaleToRange(){
	    /** Scale the incoming unsigned shorts from 0 to 65536.0 to -1 to 1.
	     * They are stored as shorts as this uses less memory and the quantization does not hurt the performance much.
	     */
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = ((m_values[i] / 65536.0) - 0.5) * 2;
		}
	}

	void fill(const T& value){
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = value;
		}
	}

	T& operator()(unsigned int i, unsigned int j, unsigned int k){
		return m_values[i * m_size * m_size + j * m_size + k];
	}

	const T& operator()(unsigned int i, unsigned int j, unsigned int k) const {
		return m_values[i * m_size * m_size + j * m_size + k];
	}

	T& operator()(const Pose3D& pose){
		return m_values[pose.convertBackToInner(m_size)];
	}

	const T& operator()(const Pose3D& pose) const {
		return m_values[pose.convertBackToInner(m_size)];
	}

    std::pair<double, double> getMinMaxValue() const {
        double minVal = DBL_MAX;
        double maxVal = -DBL_MAX;
        for(const auto& val: m_values){
            if(val > maxVal){
                maxVal = val;
            }
            if(val < minVal){
                minVal = val;
            }
        }
        return std::pair<double, double>(minVal, maxVal);
    }

	unsigned int length() const {
		return m_size;
	}
	unsigned int innerSize(){
		return m_innerSize;
	}

    void hitWallRangeInZ(int i, int j, int k, int afterHitWall, int beforeHitWall, int sphereSizeRadius, unsigned char newValue){
        if(m_spherePoses.empty()){
            for(int l = -beforeHitWall; l < afterHitWall; ++l){
                m_spherePoses.emplace_back(std::array<int, 3>{0, 0, l});
            }
            const int size = sphereSizeRadius;
            const int squaredRadius = sphereSizeRadius * sphereSizeRadius;
            for(int l = -size; l < size; ++l){
                for(int m = -size; m < size; ++m){
                    for(int n = -size; n < size; ++n){
                        const int length = l * l + m * m + n * n;
                        if(length < squaredRadius){
                            bool foundAlready = false;
                            for(const auto& pose: m_spherePoses){
                                if(pose[0] == l && pose[1] == m && pose[2] == n){
                                    foundAlready = true;
                                    break;
                                }
                            }
                            if(!foundAlready){
                                m_spherePoses.emplace_back(std::array<int, 3>{l, m, n});
                            }
                        }
                    }
                }
            }
        }
        for(const auto &pose: m_spherePoses){
            const bool check = i + pose[0] >= m_size || i + pose[0] < 0 ||
                               j + pose[1] >= m_size || j + pose[1] < 0 ||
                               k + pose[2] >= m_size || k + pose[2] < 0;
            if(!check){
                (*this)(i + pose[0], j + pose[1], k + pose[2]) = newValue;
            }
        }
    }

	void projectWallValueIntoZ(Array3D<unsigned char>& res) const {
	    /**
	     * Sets the hit the wall value for all voxels, at the first boundary.
	     * All values before that are set to FREE_VALUE.
	     */
		res.fill(LossValueType::notReachable());
		for(unsigned int i = 0; i < m_size; ++i){
			for(unsigned int j = 0; j < m_size; ++j){
                for(int k = 0; k < m_size; ++k){
				//for(int k = m_size - 1; k >= 0; --k){
					if((*this)(i, j, k) <= 0){ // collision
						// this break ensures that the values behind the first obstacle are untouched
                        res(i,j,k) = LossValueType::trueWallValue();
						break;
					}
					// free before an object was hit
					res(i,j,k) = LossValueType::freeValue();
				}
			}
		}
	}

	T& at2D(unsigned int i, unsigned int j, unsigned int value, unsigned int dim){
	    /**
	     * Treat the 3D volume as a 2D plane with different selections via the dim parameter
	     */
		if(dim == 0){
			return (*this)(value, i, j);
		}else if(dim == 1){
			return (*this)(i, value, j);
		}else{
			return (*this)(i, j, value);
		}
	}

	Pose3D findFreeElementIn(unsigned int dim, unsigned int value){
	    /**
	     * This function tries to find a free element in a 2D plane, this is useful for finding a good spot
	     * for starting the flood fill algorithm. Dim is usually 2, which corresponds to the z coordinate.
	     * As the z channel is inverted the value is usually usedSize - 1.
	     */
		if(value > m_size){
			return Pose3D(0, 0, dim, value);
		}
		double f1 = 1. / 3.;
		double f2 = 2. / 3.;
		// we already provide some locations, where it might be interesting to start, if all of them
		// fail a new position is searched
		const std::list<Pose2D> startList = {Pose2D(m_size/2,  m_size/2),
                                           Pose2D(m_size*f1, m_size*f1),
                                           Pose2D(m_size*f1, m_size*f2),
                                           Pose2D(m_size*f2, m_size*f1),
                                           Pose2D(m_size*f2, m_size*f2)};
		for(const auto& ele : startList){
		    // above zero is free
			if(at2D(ele.first, ele.second, dim, value) > 0){
				return Pose3D(ele, dim, value);
			}
		}

		// check all positions in the image to find a valid pose
		for(unsigned int i = 0; i < m_size; ++i){
			for(unsigned int j = 0; j < m_size; ++j){
				if(at2D(i,j, value, dim) > 0){
					return Pose3D(i,j, dim, value);
				}
			}
		}
		throw UnusableException();
	}

	Pose3D convertToPose(unsigned int i) const {
	    /**
	     * Converts a given index id into a 3D unsigned int position in the 3D array.
	     */
		const unsigned int first = i / (m_size * m_size);
		return {first, (unsigned int)((i - first * m_size * m_size) / m_size), (unsigned int)(i % m_size)};
	}

	void performFloodFill(const Pose3D& startPose, Array3D<unsigned char>& res){
        /**
         * This function performs the flood fill algorithm, it starts on the given start pose and writes the resulting
         * loss values in the res array.
         */

        std::vector<bool> isWallValue;
        isWallValue.resize(m_innerSize);
        std::fill(isWallValue.begin(), isWallValue.end(), false);
        std::vector<bool> isWallNotVisibleValue;
        isWallNotVisibleValue.resize(m_innerSize);
        std::fill(isWallNotVisibleValue.begin(), isWallNotVisibleValue.end(), false);
        for(unsigned int i = 0; i < m_innerSize; ++i){
            if(res.getInner(i) == LossValueType::trueWallValue()){
                isWallValue[i] = true;
            }
        }
        // we track all places we already have been too
        Array3D<unsigned int> visitedPlaces(m_size);
        std::vector<bool> visitedList;
        visitedList.resize(m_innerSize);
        int visitedCounter = 0;
        // walk over all not reachable values. At this point all values which are not directly in the line of sight
        // are NOT_REACHABLE. All other values are therefore either FREE_VALUE or are WALL_VALUE. See
        // projectWallValueIntoZ() for how this happened.
        // At the end of this for loop, we store the position of starting points for the flood fill in the visitedList.
        for(unsigned int i = 0; i < m_innerSize; ++i){
            if(res.getInner(i) == LossValueType::notReachable()){ // is behind the wall
                const auto pose = convertToPose(i);
                bool done = false;
                // now from this one position, we now check all 6 neighbouring voxels
                for(unsigned int dim = 0; dim < 3 && !done; ++dim){
                    for(int dir = -1; dir < 2; dir += 2){
                        auto newPose = pose.createAndMove(dim, dir);
                        // check if still inside of the 3D array
                        if(isInCube(newPose)){
                            if(res(newPose) == LossValueType::freeValue()){ // is free space
                                // all of these values are used to start a flood filling
                                visitedList[i] = true;
                                ++visitedCounter;
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        std::array<std::vector<bool>, 2> visitedListsArray;
        const double fac = length() / 256.;
        const int beforeHitWall = 32 * fac;
        const int afterHitWall = 16 * fac;
        const int sphereRadius = 16 * fac;

        int currentVisistedIndex = 0;
        int notCurrentVisistedIndex = 1;
        visitedListsArray[currentVisistedIndex] = visitedList;
        visitedListsArray[notCurrentVisistedIndex].resize(m_innerSize);
        // all border elements which are hidden to the camera but reachable through the flood fill
        std::vector<bool> isInnerBorder;
        isInnerBorder.resize(m_innerSize);
        unsigned int distCounter = 1;
        const unsigned int maxSeeAroundEdgesDist = MAX_SEE_AROUND_EDGES_DIST * fac;
        // we start from each marked to visit element a search for undetected elements.
        // each iteration set the distance for all current neighbours.
        while(visitedCounter > 0){
            if(distCounter > maxSeeAroundEdgesDist + 1){
                break;
            }
            std::fill(visitedListsArray[notCurrentVisistedIndex].begin(), visitedListsArray[notCurrentVisistedIndex].end(), false);
            visitedCounter = 0;
            for(unsigned int i = 0; i < m_innerSize; ++i){
                if(visitedListsArray[currentVisistedIndex][i]){
                    visitedPlaces.getInner(i) = distCounter;
                    const auto pose = convertToPose(i);
                    for(unsigned int dim = 0; dim < 3; ++dim){
                        for(int dir = -1; dir < 2; dir += 2){
                            auto newPose = pose.createAndMove(dim, dir);
                            if(isInCube(newPose)){
                                const unsigned int newIndex = newPose.convertBackToInner(m_size);
                                const unsigned char currentValue = res(newPose);
                                // not visited
                                if(!visitedList[newIndex] && currentValue == LossValueType::notReachable()){
                                    if((*this)(newPose) > 0){ // is in actual free space
                                        visitedListsArray[notCurrentVisistedIndex][newIndex] = true;
                                        visitedList[newIndex] = true;
                                        ++visitedCounter;
                                    }else{ // is not must be border
                                        isInnerBorder[newIndex] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // switch the current and not current index list
            currentVisistedIndex = (currentVisistedIndex + 1) % 2;
            notCurrentVisistedIndex = (notCurrentVisistedIndex + 1) % 2;
            // increase the dist counter
            ++distCounter;
        }
        // convert the calculated distances into loss weights
        if(distCounter > 1){
            for(unsigned int i = 0; i < m_innerSize; ++i){
                const unsigned int value = visitedPlaces.getInner(i);
                if(value > 0 && res.getInner(i) != LossValueType::trueWallValue()){
                    res.getInner(i) = LossValueType::freeNotVisible(value);
                }
            }
            // lastly add the not visible walls
            for(unsigned int i = 0; i < m_innerSize; ++i){
                if(isInnerBorder[i]){
                    const auto pose = convertToPose(i);
                    // find surrounding voxels which are also wall values, and add them too
                    for(unsigned int dim = 0; dim < 3; ++dim){
                        for(int dir = -1; dir < 2; dir += 2){
                            auto newPose = pose.createAndMove(dim, dir);
                            if(isInCube(newPose)){
                                const unsigned int newIndex = newPose.convertBackToInner(m_size);
                                // is in collision and is not already front wall
                                if((*this)(newPose) <= 0 && res.getInner(newIndex) != LossValueType::trueWallValue()){
                                    res(newPose) = LossValueType::trueWallValueNotVisible();
                                    isWallNotVisibleValue[newPose.convertBackToInner(m_size)] = true;
                                }
                            }
                        }
                    }
                    res(pose) = LossValueType::trueWallValueNotVisible();
                    isWallNotVisibleValue[pose.convertBackToInner(m_size)] = true;
                }
            }
        }


        StopWatch hitWallRange;
        for(unsigned int i = 0; i < m_innerSize; ++i){
            if(isWallNotVisibleValue[i]){
                const auto pose = convertToPose(i);
                res.hitWallRangeInZ(pose.first, pose.second, pose.third, afterHitWall, beforeHitWall, sphereRadius,
                                    LossValueType::wallValueNotVisible());
            }
        }
        for(unsigned int i = 0; i < m_innerSize; ++i){
            if(isWallValue[i]){
                const auto pose = convertToPose(i);
                res.hitWallRangeInZ(pose.first, pose.second, pose.third, afterHitWall, beforeHitWall, sphereRadius,
                                    LossValueType::wallValue());
            }
        }
        std::cout << "Time for hit wall range: " << hitWallRange.elapsed_time() << std::endl;
        for(unsigned int i = 0; i < m_innerSize; ++i){
            if(isWallNotVisibleValue[i]){
                const auto pose = convertToPose(i);
                res(pose) = LossValueType::trueWallValueNotVisible();
            }else if(isWallValue[i]){
                const auto pose = convertToPose(i);
                res(pose) = LossValueType::trueWallValue();
            }
        }
    }

	bool isInCube(const Pose3D& pose){
	    /**
	     * Checks if a given pose can be used as an index to the array
	     */
		return pose.first >= 0 && pose.first < m_size && pose.second >= 0 &&
		       pose.second < m_size && pose.third >= 0 && pose.third < m_size;
	}

	bool isInCube(const Point<int>& pose){
		/**
		 * Checks if a given pose can be used as an index to the array
		 */
		return pose[0] >= 0 && pose[0] < m_size && pose[1] >= 0 &&
			   pose[1] < m_size && pose[2] >= 0 && pose[2] < m_size;
	}

	T& getInner(unsigned int i){
		return m_values[i];
	}

private:

	std::vector<T> m_values;
	unsigned int m_innerSize;
	unsigned int m_size;

    std::vector<std::array<int, 3> > m_spherePoses;

};




#endif //LOSSCALCULATOR_ARRAY3D_H
