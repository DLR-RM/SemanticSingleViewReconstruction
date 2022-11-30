//
// Created by Dominik Winkelbauer on 27.09.18.
//

#include "Octree.h"
#include "../geom/PolygonCubeIntersection.h"
#include <map>
#include "Space.h"

const std::vector<dPoint> Octree::m_positions({{1, 1, 1}, {-1, 1, 1}, {1, -1, 1}, {-1, -1, 1}, {1, 1, -1}, {-1, 1, -1}, {1, -1, -1}, {-1, -1, -1}});
const std::vector<dPoint> Octree::m_directions({{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}});


Octree::Octree(const dPoint &origin, const dPoint &size, int level, const Octree *parent) : m_state(OCTREE_EMPTY), m_origin(origin), m_size(size), m_level(level), m_parent(parent), m_voxelAmount(0){
    m_neighbours.resize(8, nullptr);
    m_id = -1;
};

void Octree::changeState(OctreeState newState) {
    if (newState == OCTREE_EMPTY || newState == OCTREE_FILLED) {
        m_children.clear();
    } else {
        for (int i = 0; i < 8; i++) {
            m_children.emplace_back(m_origin + positionFromIndex(i) * m_size / 4, m_size / 2, m_level + 1, this);
        }

        for (int i = 0; i < 8; i++) {
            for (int d = 0; d < 6; d++) {
                dPoint neighbourPos = positionFromIndex(i) + m_directions[d] * 2;

                if (fabs(neighbourPos[0]) + fabs(neighbourPos[1]) + fabs(neighbourPos[2]) > 3) {
                    if (m_neighbours[d] != nullptr && !m_neighbours[d]->getChilren().empty()) {
                        dPoint neighbourPosInParentNeighbour = positionFromIndex(i) - m_directions[d] * 2;
                        Octree& newNeighbour = m_neighbours[d]->getChilren()[indexFromPosition(neighbourPosInParentNeighbour)];

                        m_children[i].setNeighbour(d, newNeighbour);
                        newNeighbour.setNeighbour(flipDirectionIndex(d), m_children[i]);
                    }
                } else {
                    m_children[i].setNeighbour(d, m_children[indexFromPosition(neighbourPos)]);
                }
            }
        }
    }
    m_state = newState;
}

int Octree::flipDirectionIndex(int direction) {
    if (direction % 2 == 0)
        return direction + 1;
    else
        return direction - 1;
}

int Octree::indexFromPosition(const dPoint& position) {
    for (int i = 0; i < m_positions.size(); i++) {
        if (m_positions[i] == position)
            return i;
    }
    return -1;
}

const dPoint& Octree::positionFromIndex(int index) {
    return m_positions[index];
}

std::vector<Octree> &Octree::getChilren() {
    return m_children;
}

bool Octree::intersectsWith(const Polygon &polygon) const {
    return t_c_intersection(polygon, m_origin + m_size / 2 + 1e-6, m_origin - m_size / 2 - 1e-6);
}

const std::vector<Polygon *>& Octree::getIntersectingPolygons() const {
    return m_intersectingPolygons;
}

void Octree::addIntersectingPolygon(const Polygon &polygon) {
    auto& poly = const_cast<Polygon&>(polygon);
    m_intersectingPolygons.push_back(&poly);
}

void Octree::setNeighbour(int direction, Octree &octree) {
    m_neighbours[direction] = &octree;
}

bool Octree::containsPoint(const dPoint &point) const {
    return fabs(point[0] - m_origin[0]) <= m_size[0] / 2 && fabs(point[1] - m_origin[1]) <= m_size[1] / 2 && fabs(point[2] - m_origin[2]) <= m_size[2] / 2;
}

const Octree &Octree::findNodeContainingPoint(const dPoint &point) const {
    if (m_state == OCTREE_MIXED) {
        for (auto& child: m_children) {
            if (child.containsPoint(point)) {
                return child.findNodeContainingPoint(point);
            }
        }

        printError("No child found containing given point.");
        printVar(point);
        return *this;
    } else {
        return *this;
    }
}

OctreeState Octree::getState() const {
    return m_state;
}

double Octree::calcMinDistanceToPoint(const dPoint& point) const {
    return eMaxEqual(d_zeros, eMaxEqual((m_origin - m_size / 2) - point, point - (m_origin + m_size / 2))).length();
}

double Octree::calcMaxDistanceToPoint(const dPoint& point) const {
    return eMaxEqual(d_zeros, eMaxEqual((m_origin + m_size / 2) - point, point - (m_origin - m_size / 2))).length();
}

void Octree::collectLeafChilds(std::vector<const Octree*>& childs) const  {
    if (m_state == OCTREE_MIXED) {
        for (int i = 0; i < 8; i++) {
            m_children[i].collectLeafChilds(childs);
        }
    } else {
        childs.push_back(this);
    }
}

void Octree::findLeafChildsInDirection(const dPoint& direction, std::vector<const Octree*>& neighbours) const  {
    if (m_state == OCTREE_MIXED) {
        for (int i = 0; i < 8; i++) {
            bool correctPosition = true;
            for (int j = 0; j < 3; j++) {
                if (direction[j] != m_positions[i][j] && direction[j] != 0) {
                    correctPosition = false;
                    break;
                }
            }

            if (correctPosition)
                m_children[i].findLeafChildsInDirection(direction, neighbours);
        }
    } else {
        // check if the octree is already in the list
        for(const auto* octree: neighbours){
            if(octree == this){
                return;
            }
        }
        neighbours.push_back(this);
    }
}

const Octree* Octree::getClosestNeighbour(int direction) const {
    if (m_neighbours[direction] != nullptr)
        return m_neighbours[direction];
    else if (m_parent != nullptr)
        return m_parent->getClosestNeighbour(direction);
    else
        return nullptr;
}

void Octree::findLeafNeighbours(std::vector<const Octree*>& neighbours) const {
    for (int d = 0; d < 6; d++) {
        const Octree* closestNeighbour = getClosestNeighbour(d);
        if (closestNeighbour != nullptr) {
            closestNeighbour->findLeafChildsInDirection(m_directions[d] * -1, neighbours);
        }
    }
}

int Octree::getId() const {
    return m_id;
}


const dPoint& Octree::getOrigin() const {
    return m_origin;
}


int Octree::resetIds(int nextId) {
    if (m_children.empty()) {
        m_id = nextId++;
    } else {
        for (auto& child: m_children) {
            nextId = child.resetIds(nextId);
        }
    }
    return nextId;
}

const Polygon* Octree::rayIntersectPolygon(const dPoint &start, const dPoint &end, const Polygon &startPoly) const{
    const Octree &octreeStart = findNodeContainingPoint(start);
    const Octree &octreeEnd = findNodeContainingPoint(end);
    const dPoint &startOrigin = octreeStart.getOrigin();
    const dPoint &endOrigin = octreeEnd.getOrigin();


    std::list<dPoint> neededChecks;
    if(m_voxelAmount == 0){
        printError("The voxel amount was not set!");
        exit(1);
    }
    line3D(start, end, neededChecks, m_voxelAmount);

    const Beam beam(start, end);
    const BoundingBox bigBB({-1, -1, -1}, {1, 1, 1});
    for(const auto &point: neededChecks){
        if(bigBB.isPointIn(point)){
            const Octree &currentOctree = findNodeContainingPoint(point);
            double minDist = DBL_MAX;
            const Polygon* currentPoly = nullptr;
            for(const auto *octreePoly: currentOctree.getPolygons()){
                // skip the start poly
                if(octreePoly == &startPoly){
                    continue;
                }
                if(octreePoly->rayIntersect(beam)){
                    const double newDist = octreePoly->rayIntersectDist(beam);
                    if(newDist < minDist){
                        minDist = newDist;
                        currentPoly = octreePoly;
                    }
                }
            }
            if(minDist != DBL_MAX){
                return currentPoly;
            }
        }else{
            printError("The point is not inside of the BoundingBox: " << point);
            exit(1);
        }
    }
    return nullptr;
}

bool Octree::rayIntersect(const dPoint &start, const dPoint &end, const Polygon &startPoly) const{
    const Octree& octreeStart = findNodeContainingPoint(start);
    const Octree& octreeEnd = findNodeContainingPoint(end);
    const dPoint& startOrigin = octreeStart.getOrigin();
    const dPoint& endOrigin = octreeEnd.getOrigin();


    std::list<dPoint> neededChecks;
    if(m_voxelAmount == 0){
        printError("The voxel amount was not set!");
        exit(1);
    }
    line3D(start, end, neededChecks, m_voxelAmount);

    const Beam beam(start, end);
    const BoundingBox bigBB({-1, -1, -1}, {1, 1, 1});
    for(const auto& point : neededChecks){
        if(bigBB.isPointIn(point)){
            const Octree &currentOctree = findNodeContainingPoint(point);
            for(const auto *octreePoly: currentOctree.getPolygons()){
                // skip the start poly
                if(octreePoly == &startPoly){
                    continue;
                }
                if(octreePoly->rayIntersect(beam)){
                    return true;
                }
            }
        }else{
            std::cout << point << std::endl;
            exit(1);
        }
    }
    return false;
}

void Octree::line3D(const dPoint &start, const dPoint &end, std::list<dPoint> &usedVoxels, const int voxelAmount){
    const BoundingBox bigBB({-1, -1, -1}, {1, 1, 1});
    const auto lowerGridCorner = [voxelAmount](const dPoint& point) -> dPoint {
        return (pointFloor((point + 1) * (voxelAmount / 2.0))) / (voxelAmount / 2.0) - 1.0;
    };
    // we assume the space is from -1 to 1
    const Beam beam(start, end);
    const double spaceRes = 2.0 / voxelAmount;
    dPoint dir;
    dir[0] = sigum(beam.getDir()[0]);
    dir[1] = sigum(beam.getDir()[1]);
    dir[2] = sigum(beam.getDir()[2]);
    dPoint originDir;
    originDir[0] = dir[0] == 0 ? 1 : dir[0];
    originDir[1] = dir[1] == 0 ? 1 : dir[1];
    originDir[2] = dir[2] == 0 ? 1 : dir[2];

    if(start == end){
        const dPoint currentCorner = lowerGridCorner(start);
        usedVoxels.emplace_back(currentCorner + originDir * (spaceRes * 0.5));
    }
    dPoint currentPoint = start;
    auto projectPointOntoPlaneDist = [](const dPoint& start, const dPoint& dir, const Plane& plane) -> double {
        const double divisor = dot(dir, plane.m_normal);
        if(divisor == 0.){
            return -1;
        }
        const double dividend = plane.m_dist - dot(start, plane.m_normal);
        return dividend / divisor;
    };

    // find current corner at start, this is the corner
    dPoint currentCorner = lowerGridCorner(start);
    for(int i = 0; i < 3; ++i){
        if(dir[i] < 0){
            currentCorner[i] += spaceRes;
        }
    }
    usedVoxels.emplace_back(currentCorner + originDir * (spaceRes * 0.5));


    const std::vector<dPoint> planeNormals= {{1.0,0.0,0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    // until the current point reaches the end point
    while((end - currentPoint).lengthSquared() > 1e-7){
        int minIndex = -1;
        auto minDist = DBL_MAX;
        for(int i = 0; i < 3; ++i){
            if(dir[i] == 0){
                continue;
            }
            currentCorner[i] += dir[i] * spaceRes;
            // calculate the plane in beam dir
            const Plane currentPlane(planeNormals[i], currentCorner);
            const double dist = projectPointOntoPlaneDist(currentPoint, beam.getDir(), currentPlane);
            if(minDist > dist && dist != -1){
                minDist = dist;
                minIndex = i;
            }
            currentCorner[i] -= dir[i] * spaceRes;
        }
        const double distToPoint = (currentPoint - end).length();
        //std::cout << minDist << ", " << distToPoint << ", " << currentPoint << ", " << currentCorner << ", " << spaceRes << std::endl;

        if(distToPoint < minDist){
            // the point is inside of the current voxel
            currentPoint = end;
            break;
        }else{
            // move the current corner
            currentCorner[minIndex] += dir[minIndex] * spaceRes;

            // move the current point
            currentPoint += beam.getDir() * minDist;
            // a break to make sure that it is not stuck forever, can happen if the
            if(!bigBB.isPointIn(currentPoint)){
                std::cout << start << ", " << end << ", " << beam.getDir() << ", " << dir <<  std::endl;
                exit(1);
            }

            // save the new point in the entered list
            usedVoxels.emplace_back(currentCorner + originDir * (spaceRes * 0.5));
        }
    }

}


void buildOctree(Octree& octree, int maxLevel) {
    std::map<const Polygon*, bool> assignedPolygons;
    for (auto& child : octree.getChilren()) {
        for (auto& poly : octree.getIntersectingPolygons()) {
            if (child.intersectsWith(*poly)) {
                child.addIntersectingPolygon(*poly);
                assignedPolygons[poly] = true;
            }
        }

        if (child.getIntersectingPolygons().empty()) {
            child.changeState(OCTREE_EMPTY);
        } else if (maxLevel <= 1) {
            child.changeState(OCTREE_FILLED);
        } else {
            child.changeState(OCTREE_MIXED);
            buildOctree(child, maxLevel - 1);
        }
    }

    for (const auto& poly : octree.getIntersectingPolygons()) {
        if (!assignedPolygons[poly]) {
            printError("Polygon got lost!");
            printVar(poly->getPoints()[0]);
            printVar(poly->getPoints()[1]);
            printVar(poly->getPoints()[2]);
            printVar(maxLevel);
            exit(1);
        }
    }
    octree.setVoxelAmount(int(pow(2, maxLevel)));
}

