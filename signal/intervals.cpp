#include "intervals.h"

#include <stdexcept>
#include <boost/assert.hpp>
#include <cfloat>
#include <TaskTimer.h>
#include <sstream>

#include <QtGlobal> // foreach

namespace Signal {

const IntervalType Interval::IntervalType_MIN = (IntervalType)0;
const IntervalType Interval::IntervalType_MAX = (IntervalType)-1;
const Interval Interval::Interval_ALL = Interval(Interval::IntervalType_MIN, Interval::IntervalType_MAX);
const Intervals Intervals::Intervals_ALL = Intervals(Interval::Interval_ALL);

Interval::
        Interval( IntervalType first, IntervalType last )
    :
    first(first), last(last)
{
    BOOST_ASSERT( valid() );
}


bool Interval::
        valid() const
{
    return first <= last;
}


Interval& Interval::
        operator|=(const Interval& r)
{
    first = std::min(first, r.first);
    last = std::max(last, r.last);
    return *this;
}


Interval& Interval::
        operator&=(const Interval& r)
{
    first = std::max(first, r.first);
    last = std::min(last, r.last);
    first = std::min(first, last);
    return *this;
}


bool Interval::
        operator==(const Interval& r) const
{
    return first==r.first && last==r.last;
}


bool Interval::
        operator!=(const Interval& r) const
{
    return !(*this == r);
}


Intervals::
        Intervals()
{
}


Intervals::
        Intervals(const Interval& r)
{
    if (r.count())
    {
        BOOST_ASSERT( r.valid() );
        base::push_back( r );
    }
}


Intervals::
        Intervals(IntervalType first, IntervalType last)
{
    if (first != last)
    {
        BOOST_ASSERT( first < last );
        base::push_back( Interval( first, last ) );
    }
}


Intervals& Intervals::
        operator |= (const Intervals& b)
{
    foreach (const Interval& r,  b)
        operator |= ( r );
    return *this;
}


Intervals& Intervals::
        operator |= (const Interval& r)
{
    if (0==r.count())
        return *this;

    base::iterator first = firstIntersecting( r );
    if (first==end())
    {
        base::iterator itr = base::begin();
        // find first after
        while ( itr!=base::end() && itr->first <= r.last)
            itr++;
        base::insert( itr, r );
        return *this;
    }

    base::iterator last = first;
    last++;
    // find first after
    while (last != end() && last->first <= r.last)
        last++;

    Interval b = r;

    for (base::iterator itr=first; itr!=last; itr++)
    {
        Interval& i = *itr;
        b |= i;
    }

    base::erase( first, last );
    base::insert( last, b );

    return *this;
}


Intervals& Intervals::
        operator -= (const Intervals& b)
{
    foreach (const Interval& r,  b)
        operator-=( r );
    return *this;
}


Intervals& Intervals::
        operator -= (const Interval& r)
{
    if (0==r.count())
        return *this;

    base::iterator itr = firstIntersecting( r );

    while (itr!=base::end())
    {
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
                itr = base::erase( itr );

            // Check if intersection is in the middle of 'itr'
            else if (i.first < r.first && i.last > r.last) {
                Interval j(r.last, i.last);
                itr->last = r.first;
                itr++;
                base::insert(itr, j);

            // Else, error
            } else {
                BOOST_ASSERT( false );
                throw std::logic_error("Shouldn't reach here");
            }
        } else {
            break;
        }
    }
    return *this;
}


Intervals& Intervals::
        operator >>= (const IntervalType& b)
{
    for (base::iterator itr = base::begin(); itr!=base::end();) {
        Interval& i = *itr;
	
        if (Interval::IntervalType_MIN + b > i.first ) i.first = Interval::IntervalType_MIN;
		else i.first -= b;
        if (Interval::IntervalType_MIN + b > i.last ) i.last = Interval::IntervalType_MIN;
		else i.last -= b;

        if ( Interval::IntervalType_MIN == i.first && Interval::IntervalType_MIN == i.last )
            itr = base::erase( itr );
		else
			itr++;
	}

	return *this;
}


Intervals& Intervals::
        operator <<= (const IntervalType& b)
{
    for (base::iterator itr = base::begin(); itr!=base::end();) {
        Interval& i = *itr;
	
        if (Interval::IntervalType_MAX - b <= i.first ) i.first = Interval::IntervalType_MAX;
		else i.first += b;
        if (Interval::IntervalType_MAX - b <= i.last ) i.last = Interval::IntervalType_MAX;
		else i.last += b;

        if ( Interval::IntervalType_MAX == i.first && Interval::IntervalType_MAX == i.last )
            itr = base::erase( itr );
		else
			itr++;
	}

	return *this;
}


Intervals& Intervals::
        operator &= (const Intervals& b)
{
    // TODO optimize
    Intervals rebuild;

    foreach (const Interval& r,  b) {
        Intervals copy = *this;
        copy&=( r );
        rebuild |= copy;
    }

    *this = rebuild;

    if (b.empty())
        clear();

    return *this;
}


Intervals& Intervals::
        operator &= (const Interval& r)
{
    if (0==r.count())
    {
        clear();
        return *this;
    }

    base::iterator itr = firstIntersecting( r );
    if (itr != base::begin())
        itr = base::erase(begin(), itr);

    while (itr!=base::end())
    {
        Interval& i = *itr;

        // Check if interval 'itr' does not intersect with 'r'
        if (!i.isConnectedTo(r)) {
            itr = base::erase(itr, end());

        } else {
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
        operator ^= (const Intervals& b)
{
    *this = (*this - b) | (b - *this);
    return *this;
}


Intervals& Intervals::
        operator*=(const float& scale)
{
    base::iterator itr;
    for (itr = base::begin(); itr!=base::end(); itr++) {
        itr->first*=scale;
        itr->last*=scale;
    }

    return *this;
}


Interval Intervals::
        fetchFirstInterval() const
{
    if (empty())
        return Interval( Interval::IntervalType_MIN,
                         Interval::IntervalType_MIN );

    return base::front();
}


Interval Intervals::
        fetchInterval( IntervalType dt, IntervalType center ) const
{
    if (center < dt/2)
        center = 0;
    else
        center -= dt/2;

    if (empty()) {
        return Interval( Interval::IntervalType_MIN, Interval::IntervalType_MIN );
    }

    const_iterator itr;
    for (itr = begin(); itr!=end(); itr++)
        if ( itr->first > center )
            break;

    IntervalType distance_to_next = Interval::IntervalType_MAX;
    IntervalType distance_to_prev = Interval::IntervalType_MAX;

    if (itr != end()) {
        distance_to_next = itr->first - center;
    }
    if (itr != begin()) {
        base::const_iterator itrp = itr;
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

        BOOST_ASSERT(center>=f.first);

        unsigned int_div_ceil = ( center-f.first + dt - 1 ) / dt;
        IntervalType start = f.first + dt*int_div_ceil;

        if (f.last <= start ) {
            return Interval( f.last-dt, f.last );
        }

        BOOST_ASSERT(start>=f.first);

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
    if (empty()) {
        return Interval( 0, 0 );
    }

    return Interval( base::front().first, base::back().last );
}


Intervals Intervals::
        enlarge( IntervalType dt ) const
{
    Intervals I;
    foreach (Interval r, *this)
    {
        if (r.first > dt)
            r.first -= dt;
        else
            r.first = 0;

        if (r.last < Interval::IntervalType_MAX - dt)
            r.last += dt;
        else
            r.last = Interval::IntervalType_MAX;

        I |= r;
    }
    return I;
}


Intervals Intervals::
        shrink( IntervalType dt ) const
{
    Intervals I;
    foreach (Interval r, *this)
    {
        if (r.first > 0)
        {
            if (r.first < Interval::IntervalType_MAX - dt)
                r.first += dt;
            else
                r.first = Interval::IntervalType_MAX;
        }

        if (r.last > dt)
            r.last -= dt;
        else
            r.last = 0;

        if (r.valid() && r.count())
            I |= r;
    }
    return I;
}


IntervalType Intervals::
        count() const
{
    IntervalType c = 0;

    foreach (const Interval& r, *this)
    {
        c += r.count();
    }

    return c;
}


Intervals::base::iterator Intervals::
        firstIntersecting( const Interval& b )
{
    for (base::iterator itr = base::begin(); itr!=base::end(); itr++)
        if ( itr->isConnectedTo( b ) )
            return itr;
    return base::end();
}


std::string Intervals::
        toString() const
{
    std::stringstream ss;
    ss << "{";
    if (1<size())
        ss << size() << "#";

    foreach (const Interval& r, *this)
    {
        if (1<size())
            ss << " ";
        ss << r.toString();
    }

    ss << "}";
    return ss.str();
}


std::ostream&
        operator << (std::ostream& o, const Intervals& I)
{
    return o << I.toString();
}


std::string Interval::
        toString() const
{
    std::stringstream ss;
    ss << "[" << first << ", " << last << ")";
    if (0<first)
        ss << count() << "#";
    return ss.str();
}


std::ostream&
        operator << (std::ostream& o, const Interval& I)
{
    return o << I.toString();
}


} // namespace Signal
