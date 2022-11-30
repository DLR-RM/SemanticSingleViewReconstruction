//
// Created by Dominik Winkelbauer on 27.09.18.
//

#ifndef SDFGEN_OCTREE_H
#define SDFGEN_OCTREE_H

#include <vector>
#include "../geom/Polygon.h"
#include "Array3D.h"
#include <thread>
#include <cmath>

enum OctreeState {
    OCTREE_EMPTY = 0,
    OCTREE_FILLED = 1,
    OCTREE_MIXED = 2
};

template <typename T> int sigum(T val) {
    return (T(0) < val) - (val < T(0));
}

class Octree {
public:

    Octree(const dPoint& origin, const dPoint& size, int level = 0, const Octree* parent = nullptr);

	void changeState(OctreeState newState);
	std::vector<Octree>& getChilren();
	bool intersectsWith(const Polygon& polygon) const;
	const std::vector<Polygon*>& getIntersectingPolygons() const;
	void addIntersectingPolygon(const Polygon& polygon);
	void setNeighbour(int direction, Octree& octree);
	const Octree& findNodeContainingPoint(const dPoint& point) const;
	bool containsPoint(const dPoint &point) const;
	OctreeState getState() const;
	double calcMinDistanceToPoint(const dPoint &point) const;
	double calcMaxDistanceToPoint(const dPoint &point) const;
	void findLeafNeighbours(std::vector<const Octree *> &neighbours) const;
	int resetIds(int nextId = 0);
	int getId() const;
	const dPoint& getOrigin() const;
	void collectLeafChilds(std::vector<const Octree*>& childs) const;
    void setVoxelAmount(int voxelAmount) { m_voxelAmount = voxelAmount; }

	/**
	 * Checks if there is an intersection between the start and end point, the voxel amount is used to calculate the voxel size
	 *
	 * The startPoly is ignored if an intersecting with it happens
	 * @param start start point
	 * @param end end point
	 * @param voxelAmount resolution of the octree space
	 * @param startPoly polygon which is ignored if it is between start and end
	 * @return true if a polygon is hit
	 */
	bool rayIntersect(const dPoint& start, const dPoint& end, const Polygon& startPoly) const;


    /**
     * Checks if there is an intersection between the start and end point, the voxel amount is used to calculate the voxel size
     *
     * The startPoly is ignored if an intersecting with it happens
     * @param start start point
     * @param end end point
     * @param voxelAmount resolution of the octree space
     * @param startPoly polygon which is ignored if it is between start and end
     * @return the polygon which was hit, nullptr if no polygon was hit
     */
    const Polygon* rayIntersectPolygon(const dPoint &start, const dPoint &end, const Polygon &startPoly) const;

    const Octree* m_parent;
	std::vector<Polygon*> m_intersectingPolygons;

	const std::vector<Polygon*>& getPolygons() const {
	    return m_intersectingPolygons;
	}

	/**
	 * This function calculates all intersecting voxels from a start point to an end point, the voxelamount determines the size of the voxel
	 * as the space has a size of -1 to 1, where the start and end point have to be in
	 * @param start start point of the beam
	 * @param end end point of the beam
	 * @param usedVoxels a list of voxel center points, with which the ray intersected
	 * @param voxelAmount resolution of the space
	 */
    static void line3D(const dPoint& start, const dPoint& end, std::list<dPoint>& usedVoxels, const int voxelAmount);;


	bool operator==(const Octree& other) const{
	    return other.m_id == m_id;
	}
private:
    std::vector<Octree> m_children;
    std::vector<Octree*> m_neighbours;
    OctreeState m_state;
    int m_level;
	int m_id;

    dPoint m_origin;
    dPoint m_size;
    int m_voxelAmount;

	static const std::vector<dPoint> m_positions;
	static const std::vector<dPoint> m_directions;

	int indexFromPosition(const dPoint &position);

	const dPoint& positionFromIndex(int index);

	void findLeafChildsInDirection(const dPoint &direction, std::vector<const Octree *> &neighbours) const;

	int flipDirectionIndex(int direction);
	const Octree* getClosestNeighbour(int direction) const;

};

void buildOctree(Octree& octree, int maxLevel);

#endif //SDFGEN_OCTREE_H
