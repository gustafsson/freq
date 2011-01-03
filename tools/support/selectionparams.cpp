#include "selectionparams.h"
#include <stdexcept>
#include <float.h>
#include <math.h>
#include <string>

void SelectionParams::scale(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

void SelectionParams::move(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

bool SelectionParams::getIsInside(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

void SelectionParams::range(float& start_time, float& end_time) const
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

SelectionParams::~SelectionParams(){}


RectangleSelection::RectangleSelection(Position p1, Position p2)
{
		this->p1 = p1;
		this->p2 = p2;
}

void RectangleSelection::range(float& start_time, float& end_time)
{
		start_time = std::min(p1.time, p2.time);
		end_time = std::max(p1.time, p2.time);
}

EllipseSelection::EllipseSelection(Position p1, Position p2)
{
		this->p1 = p1;
                this->p2 = p2;
}

void EllipseSelection::range(float& start_time, float& end_time)
{
		start_time = p1.time - fabs(p1.time - p2.time);
    end_time = p1.time + fabs(p1.time - p2.time);
}

void PolySelection::range(float& start_time, float& end_time)
{

}
