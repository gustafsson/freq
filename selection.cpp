#include "selection.h"
#include <stdexcept>
#include <float.h>
#include <math.h>

void Selection::scale(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

void Selection::move(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

bool Selection::getIsInside(Position p)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

void Selection::range(float& start_time, float& end_time)
{
		throw std::logic_error( std::string(__FUNCTION__) + " not implemented");
}

Selection::~Selection(){}


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