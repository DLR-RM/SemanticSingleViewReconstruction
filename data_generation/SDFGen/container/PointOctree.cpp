//
// Created by max on 14.09.20.
//

#include "PointOctree.h"

// these are correct I checked them at 2.9.2021
const std::vector<dPoint> PointOctree::m_directions({{0,  0,  1},
													 {0,  0,  -1},
													 {0,  1,  0},
													 {0,  -1, 0},
													 {0,  1,  1},
													 {0,  1,  -1},
													 {0,  -1, 1},
													 {0,  -1, -1},
													 {1,  0,  0},
													 {-1, 0,  0},
													 {1,  0,  1},
													 {1,  0,  -1},
													 {-1, 0,  1},
													 {-1, 0,  -1},
													 {1,  1,  0},
													 {1,  -1, 0},
													 {-1, 1,  0},
													 {-1, -1, 0},
													 {1,  1,  1},
													 {1,  1,  -1},
													 {1,  -1, 1},
													 {1,  -1, -1},
													 {-1, 1,  1},
													 {-1, 1,  -1},
													 {-1, -1, 1},
													 {-1, -1, -1}});

const BoundingBox PointOctree::m_bigBB({-1, -1, -1}, {1, 1, 1});

double PointOctree::getDistanceToOctreeSideWith(const dPoint& point, const dPoint& origin,
												const dPoint& size){
	auto minDist = DBL_MAX;
	const dPoint centerPoint = origin + size * 0.5;
	const dPoint halfSize = size * 0.5;
	for(int i = 0; i < 3; ++i){
		const double d = (centerPoint[i] + halfSize[i]) - point[i];
		if(d < minDist){
			minDist = d;
		}
		const double dNeg = point[i] - (centerPoint[i] - halfSize[i]);
		if(dNeg < minDist){
			minDist = dNeg;
		}
	}
	return minDist;
}

double PointOctree::getDistanceToOctreeSide(const dPoint& point) const{
	/**
	 * Calculate the shortest distance of the given point to one of the sides of the cube along the axis.
	 */
	return getDistanceToOctreeSideWith(point, m_origin, m_size);
}

void PointOctree::findExternalNeighbors(PointOctree& currentOctree) const{
	/**
	 * Find the neighbors neighbors nodes, this is necessary as the truncation threshold is 0.1 the distance of points to the
	 * surface is 2 * 0.1, while the size of a cube is only 0.125
	 */
	if(!currentOctree.m_checkedForExternalNeighbors){
		// make sure all neighbors of the external neigbors are also set
		for(const auto& neighbor: currentOctree.m_neighbors){
			PointOctree& neigborNotConst = const_cast<PointOctree&>(*neighbor);
			findNeighbors(neigborNotConst);
		}
		// walk over all neighbors
		for(const auto& neighbor: currentOctree.m_neighbors){
			// check for each of their external neighbors
			for(const auto& externalNeighbor: neighbor->m_neighbors){
				if(externalNeighbor == &currentOctree || externalNeighbor->m_state == OCTREE_EMPTY){
					continue;
				}
				// make sure that the element is not already in the external neighbor list
				bool isInList = false;
				for(const auto& inListNeighbor: currentOctree.m_externalNeighbors){
					if(inListNeighbor == externalNeighbor){
						isInList = true;
						break;
					}
				}
				if(!isInList){
					// make sure that the element is not in the neighbor list as well, as that one has already been checked
					for(const auto& inListNeighbor: currentOctree.m_neighbors){
						if(inListNeighbor == externalNeighbor){
							isInList = true;
							break;
						}
					}
					// if it is not in any list add it to the external list
					if(!isInList){
						currentOctree.m_externalNeighbors.emplace_back(externalNeighbor);
					}
				}
			}
		}
		currentOctree.m_checkedForExternalNeighbors = true;
	}
}

void PointOctree::findNeighbors(PointOctree& currentOctree) const{
	/**
	 * This finds the neighbors of the given octree, this has to be executed for the top most octree
	 */
	if(!currentOctree.m_checkedForNeighbors){
		// find neighbors if they are not set yet
		const dPoint centerPoint = currentOctree.m_origin + currentOctree.m_size * 0.5;
		for(const auto& dir: m_directions){
			const dPoint newPoint = centerPoint + dir * currentOctree.m_size[0];
			if(m_bigBB.isPointIn(newPoint)){
				const auto& currentNeighbor = findNodeContainingPoint(newPoint);
				currentOctree.m_neighbors.emplace_back(&currentNeighbor);
			}
		}
		currentOctree.m_checkedForNeighbors = true;
	}
}

PointOctree& PointOctree::findNodeContainingPoint(const dPoint& point) const{
	/**
	 * Finds the octree which contains this point, this octree will be a leaf
	 */
	if(m_state == OCTREE_MIXED){
		for(auto& child: m_children){
			if(child.containsPoint(point)){
				return child.findNodeContainingPoint(point);
			}
		}

		printError("No child found containing given point.");
		printVar(point);
		return const_cast<PointOctree&>(*this);
	}else if(m_state == OCTREE_EMPTY || m_state == OCTREE_FILLED){
		return const_cast<PointOctree&>(*this);
	}else{
		printError("Something went wrong the state is incorrect!");
		return const_cast<PointOctree&>(*this);
	}
}

bool PointOctree::containsPoint(const dPoint& point, const dPoint& origin, const dPoint& size){
	/**
	 * Checks if a given point is inside the cube spanned by the origin and the size, the origin lies in the lower left corner
	 * The size goes from the lower left corner to the upper right corner on the opposite side of the cube.
	 */
	return origin[0] <= point[0] && point[0] < origin[0] + size[0] && origin[1] <= point[1] &&
		   point[1] < origin[1] + size[1] && origin[2] <= point[2] &&
		   point[2] < origin[2] + size[2];

}

bool PointOctree::containsPoint(const dPoint& point) const{
	/**
	 * Checks if a given point is inside the cube spanned by the origin and the size, the origin of the current octree
	 */
	return containsPoint(point, m_origin, m_size);
}

PointOctree::PointOctree(std::vector<DistPoint>&& points, int maxLevel, const dPoint size,
						 dPoint origin, int level): m_size(size),
													m_origin(origin),
													m_checkedForNeighbors(false),
													m_checkedForExternalNeighbors(false){
	/**
	 * Creates a point octree, the origin lies in the lower left corner of each voxel, the size forms the end point:
	 *  end_point = size + origin
	 * The given points are only used if the maxLevel is reached else a sub octrees are created and the points are moved there.
	 * The highest Point Octree should have -1,-1,-1 to 1,1,1 as a bounding box else the m_bigBB variable has to be redefined.
	 */
	if(level == maxLevel){
		if(points.empty()){
			m_state = OCTREE_EMPTY;
		}else{
			m_state = OCTREE_FILLED;
			m_points = std::move(points);
		}
	}else{
		// split up the points
		m_state = OCTREE_MIXED;
		const dPoint sizeHalf = size * 0.5;
		const dPoint orgOrigin = origin;
		for(int i = 0; i < 2; ++i){
			origin[0] = orgOrigin[0] + (sizeHalf[0] * i);
			for(int j = 0; j < 2; ++j){
				origin[1] = orgOrigin[1] + (sizeHalf[1] * j);
				for(int k = 0; k < 2; ++k){
					origin[2] = orgOrigin[2] + (sizeHalf[2] * k);
					std::vector<DistPoint> currentPoints;
					for(const auto& point: points){
						if(containsPoint(point, origin, sizeHalf)){
							currentPoints.emplace_back(point);
						}
					}
					// this constructs even empty nodes, that is easier than to check if the used nodes exist
					m_children.emplace_back(std::move(currentPoints), maxLevel, sizeHalf, origin,
											level + 1);
				}
			}
		}
	}
}

PointDist PointOctree::distance(const dPoint& point) const{
	auto& currentOctree = findNodeContainingPoint(point);
	findNeighbors(currentOctree);
	PointDist closestPoint(DBL_MAX, nullptr);
	currentOctree.distanceToPoints(point, closestPoint);
	if(closestPoint.second != nullptr){
		// this dist is squared as the closestPoint.first distance is also squared
		const double distToOctreeSide = currentOctree.getDistanceToOctreeSide(point);
		if(closestPoint.first < distToOctreeSide * distToOctreeSide){
			// the closest point is closer than any of the six sides of the cube
			closestPoint.first = sqrt(closestPoint.first);
			return closestPoint;
		}
	}
	for(const auto& neighbor: currentOctree.m_neighbors){
		neighbor->distanceToPoints(point, closestPoint);
	}
	if(closestPoint.second != nullptr){
		const double distToBigOctreeSide = getDistanceToOctreeSideWith(point,
																	   currentOctree.m_origin -
																	   currentOctree.m_size,
																	   currentOctree.m_size * 3);
		if(closestPoint.first < distToBigOctreeSide * distToBigOctreeSide){
			// the closest point is closer than any of the six sides of the cube neighbor cubes
			closestPoint.first = sqrt(closestPoint.first);
			return closestPoint;
		}
	}
	findExternalNeighbors(currentOctree);
	for(const auto& neighbor: currentOctree.m_externalNeighbors){
		neighbor->distanceToPoints(point, closestPoint);
	}

	if(closestPoint.second != nullptr){
		closestPoint.first = sqrt(closestPoint.first);
	}
	return closestPoint;
}

void PointOctree::distanceToPoints(const dPoint& point, PointDist& pointDist) const{
	/**
	 * Calculates the closest distances to the given point for all points saved in this octree.
	 */
	for(const auto& p: m_points){
		const double dist = (p - point).lengthSquared();
		if(pointDist.first > dist){
			pointDist.first = dist;
			pointDist.second = &p;
		}
	}
}
