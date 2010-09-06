#include "intervals.h"

#include <stdexcept>
#include <boost/foreach.hpp>
#include <boost/assert.hpp>
#include <cfloat>
#include <TaskTimer.h>
#include <sstream>

namespace Signal {

const IntervalType IntervalType_MIN = (IntervalType)0;
const IntervalType IntervalType_MAX = (IntervalType)-1;
const Intervals Intervals_ALL = Intervals(IntervalType_MIN, IntervalType_MAX);
const Interval Interval_ALL = { IntervalType_MIN, IntervalType_MAX };

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
        Intervals(Interval r)
{
    BOOST_ASSERT( r.first < r.last );
    _intervals.push_back( r );
}

Intervals::
        Intervals(IntervalType first, IntervalType last)
{
    BOOST_ASSERT( first < last );
    Interval r = { first, last };
    _intervals.push_back( r );
}

Intervals& Intervals::
        operator |= (const Intervals& b)
{
    BOOST_FOREACH (const Interval& r,  b._intervals)
        operator|=( r );
    return *this;
}

Intervals& Intervals::
        operator |= (const Interval& r)
{
    _intervals.push_back( r );
    _intervals.sort();

    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end(); ) {
        std::list<Interval>::iterator next = itr;
        next++;
        if (next!=_intervals.end()) {
            Interval& a = *itr;
            Interval& b = *next;

            if (a.isConnectedTo(b))
            {
                a |= b;
                itr = _intervals.erase( next );
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
    BOOST_FOREACH (const Interval& r,  b._intervals)
        operator-=( r );
    return *this;
}

Intervals& Intervals::
        operator -= (const Interval& r)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
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
                itr = _intervals.erase( itr );

            // Check if intersection is in the middle of 'itr'
            else if (i.first < r.first && i.last > r.last) {
                Interval j = {r.last, i.last};
                itr->last = r.first;
                itr++;
                _intervals.insert(itr, j);
//                itr++;

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
        operator -= (const IntervalType& b)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
        Interval& i = *itr;
	
        if (IntervalType_MIN + b > i.first ) i.first = IntervalType_MIN;
		else i.first -= b;
        if (IntervalType_MIN + b > i.last ) i.last = IntervalType_MIN;
		else i.last -= b;

        if ( IntervalType_MIN == i.first && IntervalType_MIN == i.last )
			itr = _intervals.erase( itr );
		else
			itr++;
	}

	return *this;
}

Intervals& Intervals::
        operator += (const IntervalType& b)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
        Interval& i = *itr;
	
        if (IntervalType_MAX - b < i.first ) i.first = IntervalType_MAX;
		else i.first += b;
        if (IntervalType_MAX - b < i.last ) i.last = IntervalType_MAX;
		else i.last += b;

        if ( IntervalType_MAX == i.first && IntervalType_MAX == i.last )
			itr = _intervals.erase( itr );
		else
			itr++;
	}

	return *this;
}

Intervals& Intervals::
        operator &= (const Intervals& b)
{
	Intervals rebuild;
	BOOST_FOREACH (const Interval& r,  b._intervals) {
		Intervals copy = *this;
        copy&=( r );
		rebuild |= copy;
	}

	this->_intervals = rebuild._intervals;

	if (b._intervals.empty())
		_intervals.clear();

	return *this;
}

Intervals& Intervals::
        operator &= (const Interval& r)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
        Interval& i = *itr;

        // Check if interval 'itr' does not intersect with 'r'
        if ((i.last<=r.first) || (r.last<=i.first)) {
            itr = _intervals.erase(itr);

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
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        itr->first*=scale;
        itr->last*=scale;
    }

    return *this;
}

Interval Intervals::
        getInterval( IntervalType dt, IntervalType center ) const
{
    if (0 == _intervals.size()) {
        Interval r = {IntervalType_MIN, IntervalType_MIN};
        return r;
    }

    std::list<Interval>::const_iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        if (itr->first >= center)
            break;
    }

    IntervalType distance_to_next = IntervalType_MAX;
    IntervalType distance_to_prev = IntervalType_MAX;

    if (itr != _intervals.end()) {
        distance_to_next = itr->first - center;
    }
    if (itr != _intervals.begin()) {
        std::list<Interval>::const_iterator itrp = itr;
        itrp--;
        if (itrp->last < center )
            distance_to_prev = center - itrp->last;
        else
            distance_to_prev = 0;
    }
    if (distance_to_next<=distance_to_prev) {
        const Interval &f = *itr;
        if (f.last - f.first < dt ) {
            Interval r = f;
            return r;
        }
        Interval r = { f.first, f.first + dt };
        return r;
    } else { // distance_to_next>distance_to_prev
        itr--; // get previous Interval
        const Interval &f = *itr;
        if (f.last - f.first < dt ) {
            Interval r = f;
            return r;
        }

        if (f.last <= center ) {
            Interval r = { f.last-dt, f.last };
            return r;
        }

        IntervalType start = f.first + dt*(unsigned)((center-f.first) / dt);
        Interval r = {start, std::min(start+dt, f.last) };
        return r;
    }
}


Intervals Intervals::
        inverse() const
{
    return Intervals_ALL - *this;
}


Interval Intervals::
        getInterval( Interval n ) const
{
    Intervals sid = *this & n;

    if (0 == sid._intervals.size()) {
        Interval r = {IntervalType_MIN, IntervalType_MIN};
        return r;
    }

    return sid.intervals().front();
}

Interval Intervals::
        coveredInterval() const
{
    Interval i;
    if (_intervals.empty()) {
        i.first = i.last = 0;
        return i;
    }

    i.first = _intervals.front().first;
    i.last = _intervals.back().last;

    return i;
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

std::ostream& operator<<( std::ostream& s, const Intervals& i)
{
    s << "{" << i.intervals().size() << " interval" << ((i.intervals().size()==1)?"":"s");

    BOOST_FOREACH (const Interval& r, i.intervals())
        s << " " << r;

    return s << "}";
}

std::ostream& operator<<( std::ostream& s, const Interval& i)
{
    return s << "[" << i.first << ", " << i.last << "]";
}

} // namespace Signal
