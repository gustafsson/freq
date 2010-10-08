#include "intervals.h"

#include <stdexcept>
#include <boost/foreach.hpp>
#include <boost/assert.hpp>
#include <cfloat>
#include <TaskTimer.h>
#include <sstream>

namespace Signal {

const IntervalType Interval::Interval::IntervalType_MIN = (IntervalType)0;
const IntervalType Interval::Interval::IntervalType_MAX = (IntervalType)-1;
const Interval Interval::Interval_ALL = Interval(Interval::IntervalType_MIN, Interval::IntervalType_MAX);
const Intervals Intervals::Intervals_ALL = Intervals(Interval::Interval_ALL);

bool Interval::
        valid() const
{
    return first < last;
}


bool Interval::
        isConnectedTo(const Interval& r) const
{
    return last >= r.first && r.last >= first;
}


bool Interval::
        operator<(const Interval& r) const
{
    return first < r.first;
}


Interval& Interval::
        operator|=(const Interval& r)
{
    first = std::min(first, r.first);
    last = std::max(last, r.last);
    return *this;
}


bool Interval::
        operator==(const Interval& r) const
{
    return first==r.first && last==r.last;
}


Intervals::
        Intervals()
{
}


Intervals::
        Intervals(const Interval& r)
{
    BOOST_ASSERT( r.first < r.last );
    this->push_back( r );
}


Intervals::
        Intervals(IntervalType first, IntervalType last)
{
    BOOST_ASSERT( first < last );
    this->push_back( Interval( first, last ) );
}


Intervals& Intervals::
        operator |= (const Intervals& b)
{
    BOOST_FOREACH (const Interval& r,  b)
        operator|=( r );
    return *this;
}


Intervals& Intervals::
        operator |= (const Interval& r)
{
    this->push_back( r );
    this->sort();

    for (std::list<Interval>::iterator itr = this->begin(); itr!=this->end(); ) {
        std::list<Interval>::iterator next = itr;
        next++;
        if (next!=this->end()) {
            Interval& a = *itr;
            Interval& b = *next;

            if (a.isConnectedTo(b))
            {
                a |= b;
                itr = this->erase( next );
                itr--;
            } else {
                itr++;
            }
        } else {
            itr++;
        }
    }
    return *this;
}


Intervals& Intervals::
        operator -= (const Intervals& b)
{
    BOOST_FOREACH (const Interval& r,  b)
        operator-=( r );
    return *this;
}


Intervals& Intervals::
        operator -= (const Interval& r)
{
    for (std::list<Interval>::iterator itr = this->begin(); itr!=this->end();) {
        Interval& i = *itr;
        // Check if interval 'itr' intersects with 'r'
        if (i.isConnectedTo(r)) {

            // Check if intersection is over the start of 'itr'
            if (i.first >= r.first && i.last > r.last) {
                i.first = r.last;
                itr++;
            }

            // Check if intersection is over the end of 'itr'
            else if (i.first < r.first && i.last <= r.last) {
                i.last = r.first;
                itr++;
            }

            // Check if intersection is over the entire 'itr'
            else if (i.first >= r.first && i.last <= r.last)
                itr = this->erase( itr );

            // Check if intersection is in the middle of 'itr'
            else if (i.first < r.first && i.last > r.last) {
                Interval j(r.last, i.last);
                itr->last = r.first;
                itr++;
                this->insert(itr, j);

            // Else, error
            } else {
                BOOST_ASSERT( false );
                throw std::logic_error("Shouldn't reach here");
            }
        } else {
            itr++;
        }
    }
    return *this;
}


Intervals& Intervals::
        operator >>= (const IntervalType& b)
{
    for (std::list<Interval>::iterator itr = this->begin(); itr!=this->end();) {
        Interval& i = *itr;
	
        if (Interval::IntervalType_MIN + b > i.first ) i.first = Interval::Interval::IntervalType_MIN;
		else i.first -= b;
        if (Interval::IntervalType_MIN + b > i.last ) i.last = Interval::IntervalType_MIN;
		else i.last -= b;

        if ( Interval::IntervalType_MIN == i.first && Interval::IntervalType_MIN == i.last )
            itr = this->erase( itr );
		else
			itr++;
	}

	return *this;
}


Intervals& Intervals::
        operator <<= (const IntervalType& b)
{
    for (std::list<Interval>::iterator itr = this->begin(); itr!=this->end();) {
        Interval& i = *itr;
	
        if (Interval::IntervalType_MAX - b < i.first ) i.first = Interval::IntervalType_MAX;
		else i.first += b;
        if (Interval::IntervalType_MAX - b < i.last ) i.last = Interval::IntervalType_MAX;
		else i.last += b;

        if ( Interval::IntervalType_MAX == i.first && Interval::IntervalType_MAX == i.last )
            itr = this->erase( itr );
		else
			itr++;
	}

	return *this;
}


Intervals& Intervals::
        operator &= (const Intervals& b)
{
	Intervals rebuild;
    BOOST_FOREACH (const Interval& r,  b) {
		Intervals copy = *this;
        copy&=( r );
		rebuild |= copy;
	}

    *this = rebuild;

    if (b.empty())
        this->clear();

	return *this;
}


Intervals& Intervals::
        operator &= (const Interval& r)
{
    for (std::list<Interval>::iterator itr = this->begin(); itr!=this->end();) {
        Interval& i = *itr;

        // Check if interval 'itr' does not intersect with 'r'
        if ((i.last<=r.first) || (r.last<=i.first)) {
            itr = this->erase(itr);

        } else {
            BOOST_ASSERT(i.isConnectedTo(r));

            // Check if intersection is over the start of 'itr'
            if (i.first >= r.first && i.last > r.last)
                i.last = r.last;

            // Check if intersection is over the end of 'itr'
            else if (i.first < r.first && i.last <= r.last)
                i.first = r.first;

            // Check if intersection is over the entire 'itr'
            else if (i.first >= r.first && i.last <= r.last)
            {}

            // Check if intersection is in the middle of 'itr'
            else if (i.first < r.first && i.last > r.last) {
                i.first = r.first;
                i.last = r.last;

            // Else, error
            } else {
                throw std::logic_error("Shouldn't reach here");
            }
            itr++;
        }
    }
    return *this;
}


Intervals& Intervals::
        operator*=(const float& scale)
{
    std::list<Interval>::iterator itr;
    for (itr = this->begin(); itr!=this->end(); itr++) {
        itr->first*=scale;
        itr->last*=scale;
    }

    return *this;
}


Interval Intervals::
        getInterval() const
{
    if (this->empty())
        return Interval( Interval::IntervalType_MIN,
                         Interval::IntervalType_MIN );

    return this->front();
}


Interval Intervals::
        getInterval( IntervalType dt, IntervalType center ) const
{
    if (0 == this->size()) {
        return Interval( Interval::IntervalType_MIN, Interval::IntervalType_MIN );
    }

    std::list<Interval>::const_iterator itr;
    for (itr = this->begin(); itr!=this->end(); itr++) {
        if (itr->first >= center)
            break;
    }

    IntervalType distance_to_next = Interval::IntervalType_MAX;
    IntervalType distance_to_prev = Interval::IntervalType_MAX;

    if (itr != this->end()) {
        distance_to_next = itr->first - center;
    }
    if (itr != this->begin()) {
        std::list<Interval>::const_iterator itrp = itr;
        itrp--;
        if (itrp->last < center )
            distance_to_prev = center - itrp->last;
        else
            distance_to_prev = 0;
    }
    if (distance_to_next<=distance_to_prev) {
        const Interval &f = *itr;
        if (f.count() < dt ) {
            return f;
        }
        return Interval( f.first, f.first + dt );

    } else { // distance_to_next>distance_to_prev
        itr--; // get previous Interval
        const Interval &f = *itr;
        if (f.count() < dt ) {
            return f;
        }

        if (f.last <= center ) {
            return Interval( f.last-dt, f.last );
        }

        IntervalType start = f.first + dt*(unsigned)((center-f.first) / dt);
        return Interval( start, std::min(start+dt, f.last) );
    }
}


Intervals Intervals::
        inverse() const
{
    return Intervals_ALL - *this;
}


Interval Intervals::
        coveredInterval() const
{
    if (this->empty()) {
        return Interval( 0, 0 );
    }

    return Interval( this->front().first, this->back().last );
}


void Intervals::
        print( std::string title ) const
{
    std::stringstream ss;
    ss << *this;

    TaskTimer("%s, %s",
              title.empty()?"SamplesIntervalDescriptor":title.c_str(),
              ss.str().c_str()).suppressTiming();
}


std::ostream& operator<<( std::ostream& s, const Intervals& I)
{
    s << "{" << I.size() << " interval" << ((I.size()==1)?"":"s");

    BOOST_FOREACH (const Interval& r, I)
        s << " " << r;

    return s << "}";
}


std::ostream& operator<<( std::ostream& s, const Interval& i)
{
    return s << "[" << i.first << ", " << i.last << "]";
}

} // namespace Signal
