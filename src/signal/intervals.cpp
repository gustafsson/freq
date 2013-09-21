#include "intervals.h"

#include "exceptionassert.h"
#include "TaskTimer.h"

#include <stdexcept>
#include <cfloat>
#include <sstream>
#include <limits.h>

#include <boost/foreach.hpp>

namespace Signal {

// long unsigned
//const IntervalType Interval::IntervalType_MIN = (IntervalType)0;
//const IntervalType Interval::IntervalType_MAX = (IntervalType)-1;

// long long
const IntervalType Interval::IntervalType_MIN = LLONG_MIN;
const IntervalType Interval::IntervalType_MAX = LLONG_MAX;

const Interval Interval::Interval_ALL = Interval(Interval::IntervalType_MIN, Interval::IntervalType_MAX);
const Intervals Intervals::Intervals_ALL = Intervals(Interval::Interval_ALL);

Interval::
        Interval()
    :
    first(0),last(0)
{}


Interval::
        Interval( IntervalType first, IntervalType last )
    :
    first(first), last(last)
{
    if (!valid())
        EXCEPTION_ASSERT( valid() );
}


Interval Interval::
        spanned(const Interval& r) const
{
    return Interval(std::min(first, r.first), std::max(last, r.last));
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
    return valid() && r.valid() && ((first==r.first && last==r.last) || (r.count()==0 && count()==0));
}


bool Interval::
        operator!=(const Interval& r) const
{
    return !(*this == r);
}


bool Interval::
        contains (const Interval& t) const
{
    return first <= t.first && last >= t.last;
}


bool Interval::
        contains (const IntervalType& t) const
{
    return t >= first && t < last;
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
        base::push_back( r );
    }
}


Intervals::
        Intervals(IntervalType first, IntervalType last)
{
    if (first != last)
    {
        EXCEPTION_ASSERT( first < last );
        base::push_back( Interval( first, last ) );
    }
}


Intervals& Intervals::
        operator |= (const Intervals& b)
{
    BOOST_FOREACH (const Interval& r, b)
        operator |= ( r );
    return *this;
}


Intervals& Intervals::
        operator |= (const Interval& r)
{
    if (0==r.count())
        return *this;

    base::iterator first = base::end();
    for (base::iterator itr = base::begin(); itr!=base::end(); itr++)
        if ( r.first <= itr->last && itr->first <= r.last )
        {
            first = itr;
            break;
        }

    if (first==base::end())
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
    while (last != base::end() && last->first <= r.last)
        last++;

    Interval b = r;

    for (base::iterator itr=first; itr!=last; itr++)
    {
        b = b.spanned(*itr);
    }

    base::erase( first, last );
    base::insert( last, b );

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
    if (0==r.count())
        return *this;

    base::iterator itr = firstIntersecting( r );

    while (itr!=base::end())
    {
        Interval& i = *itr;
        // Check if interval 'itr' intersects with 'r'
        if ((i & r).count()) {

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
                EXCEPTION_ASSERT( false );
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

    BOOST_FOREACH (const Interval& r,  b) {
        Intervals copy = *this;
        copy &= r;
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
        itr = base::erase(base::begin(), itr);

    while (itr!=base::end())
    {
        Interval& i = *itr;

        // Check if interval 'itr' does not intersect with 'r'
        if (0 == (i & r).count()) {
            itr = base::erase(itr, base::end());

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
            else if (i.first < r.first && i.last > r.last)
            {
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


bool Intervals::
        contains    (const Intervals& t) const
{
    return (*this & t) == t;
}


bool Intervals::
        contains    (const Interval& t) const
{
    return (*this & t) == t;
}


bool Intervals::
        contains    (const IntervalType& t) const
{
    if (t >= Interval::IntervalType_MAX)
        return false;

    return contains(Interval(t, t+1));
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
        fetchInterval( UnsignedIntervalType dt, IntervalType center ) const
{
    if (center < IntervalType(dt/2))
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

    UnsignedIntervalType distance_to_next = Interval::IntervalType_MAX;
    UnsignedIntervalType distance_to_prev = Interval::IntervalType_MAX;

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

        EXCEPTION_ASSERT(center>=f.first);

        unsigned int_div_ceil = ( center-f.first + dt - 1 ) / dt;
        IntervalType start = f.first + dt*int_div_ceil;

        if (f.last <= start ) {
            return Interval( IntervalType(f.last-dt), f.last );
        }

        EXCEPTION_ASSERT(start>=f.first);

        return Interval( start, std::min(IntervalType(start+dt), f.last) );
    }
}


Intervals Intervals::
        inverse() const
{
    return Intervals_ALL - *this;
}


Interval Intervals::
        spannedInterval() const
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
    BOOST_FOREACH (Interval r, *this)
    {
        if (r.first > Interval::IntervalType_MIN + dt)
            r.first -= dt;
        else
            r.first = Interval::IntervalType_MIN;

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
    BOOST_FOREACH (Interval r, *this)
    {
        if (r.first > Interval::IntervalType_MIN)
        {
            if (r.first < Interval::IntervalType_MAX - dt)
                r.first += dt;
            else
                r.first = Interval::IntervalType_MAX;
        }

        if (r.last > Interval::IntervalType_MIN + dt)
            r.last -= dt;
        else
            r.last = Interval::IntervalType_MIN;

        if (r.valid() && r.count())
            I |= r;
    }
    return I;
}


IntervalType Intervals::
        count() const
{
    IntervalType c = 0;

    BOOST_FOREACH (const Interval& r, *this)
    {
        c += r.count();
    }

    return c;
}


bool Intervals::
        testSample( IntervalType const& p ) const
{
    return *this & Interval( p, p+1 );
}


Intervals::base::iterator Intervals::
        firstIntersecting( const Interval& b )
{
    for (base::iterator itr = base::begin(); itr!=base::end(); itr++)
        if ( (*itr & b).count() )
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

    BOOST_FOREACH (const Interval& r, *this)
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

    ss << "[";

    if (first != IntervalType_MIN)
        ss << first;
    else
        ss << "-";

    ss << ", ";

    if (last != IntervalType_MAX)
        ss << last;
    else
        ss << "+";

    ss << ")";

    if (0 != first && first != IntervalType_MIN && last != IntervalType_MAX)
        ss << count() << "#";

    return ss.str();
}


std::ostream&
        operator << (std::ostream& o, const Interval& I)
{
    return o << I.toString();
}


Intervals  operator |  (const Interval& a, const Intervals& b) { return Intervals(a)|=b; }
Intervals  operator -  (const Interval& a, const Intervals& b) { return Intervals(a)-=b; }
Intervals  operator &  (const Interval& a, const Intervals& b) { return Intervals(a)&=b; }
Intervals  operator ^  (const Interval& a, const Intervals& b) { return Intervals(a)^=b; }
Intervals  operator |  (const Interval& a, const Interval& b)  { return a|Intervals(b); }

} // namespace Signal

#include "timer.h"
#include "exceptionassert.h"

namespace Signal {

void Intervals::
        test()
{
    // It should be fast
    {
        const int N = 1000000;
        Intervals I;
        Timer t;
        for (int i=0; i<N; ++i) {
            I |= Interval(i,i+1);
        }
        double T = t.elapsed ()/N;
        EXCEPTION_ASSERT_LESS(T,0.0000005);
        EXCEPTION_ASSERT_EQUALS(I, Intervals(0,N));

        I = Intervals(0,N);
        t.restart ();
        for (int i=0; i<N; ++i) {
            (I & Interval(i,i+1));
        }
        T = t.elapsed ()/N;
        EXCEPTION_ASSERT_LESS(T,0.000002);
    }
}


} // namespace Signal
