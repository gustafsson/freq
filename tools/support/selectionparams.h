// TODO use these
#include "heightmap/position.h"
#include <vector>
#include <boost/shared_ptr.hpp>

// TODO move to first line of file
#ifndef SELECTION_H
#define SELECTION_H

// TODO move to after definition of class Selection
typedef boost::shared_ptr<class Selection> pSelection;

// TODO never use "using namespace" in header file
using namespace Heightmap;

// TODO use
// namespace Tools {
//    namespace Support {

class SelectionParams
{
public:
	bool inverted;
	virtual bool getIsInside(Position p);
	
	//Standard manipulation of the selection
	virtual void move(Position p);
	virtual void scale(Position p);
        virtual void range(float& start_time, float& end_time) const;
	
    virtual ~SelectionParams();
};


class RectangleSelection: public SelectionParams
{
public:
	Position p1, p2;
	
	RectangleSelection(Position p1, Position p2);
	
	void range(float& start_time, float& end_time);
};


class EllipseSelection: public SelectionParams
{
public:
	Position p1, p2;
	
	EllipseSelection(Position p1, Position p2);
	
	void range(float& start_time, float& end_time);
};


class PolySelection: public SelectionParams
{
public:
	std::vector<Position> point;
	
	void range(float& start_time, float& end_time);
};


class SplineSelection: public PolySelection
{};

// TODO end files with #endif // SELECTION_H
#endif
