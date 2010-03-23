#include "position.h"
#include <vector>

#ifndef SELECTION_H
#define SELECTION_H

typedef boost::shared_ptr<class Selection> pSelection;

class Selection
{
public:
	bool inverted;
	virtual bool getIsInside(Position p);
	
	//Standard manipulation of the selection
	virtual void move(Position p);
	virtual void scale(Position p);
	virtual void range(float& start_time, float& end_time);
	
	virtual ~Selection();
};


class RectangleSelection: public Selection
{
public:
	Position p1, p2;
	
	RectangleSelection(Position p1, Position p2);
	
	void range(float& start_time, float& end_time);
};


class EllipseSelection: public Selection
{
public:
	Position p1, p2;
	
	EllipseSelection(Position p1, Position p2);
	
	void range(float& start_time, float& end_time);
};


class PolySelection: public Selection
{
public:
	std::vector<Position> point;
	
	void range(float& start_time, float& end_time);
};


class SplineSelection: public PolySelection
{};

#endif