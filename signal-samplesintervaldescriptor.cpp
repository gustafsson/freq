#include "signal-samplesintervaldescriptor.h"
#include <stdexcept>
#include <boost/foreach.hpp>
#include <boost/assert.hpp>
#include <cfloat>
#include <TaskTimer.h>
#include <sstream>

namespace Signal {

const SamplesIntervalDescriptor::SampleType SamplesIntervalDescriptor::SampleType_MIN = (SamplesIntervalDescriptor::SampleType)0;
const SamplesIntervalDescriptor::SampleType SamplesIntervalDescriptor::SampleType_MAX = (SamplesIntervalDescriptor::SampleType)-1;
const SamplesIntervalDescriptor SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL = SamplesIntervalDescriptor(SamplesIntervalDescriptor::SampleType_MIN, SamplesIntervalDescriptor::SampleType_MAX);
const SamplesIntervalDescriptor::Interval SamplesIntervalDescriptor::SamplesInterval_ALL = { SamplesIntervalDescriptor::SampleType_MIN, SamplesIntervalDescriptor::SampleType_MAX };

bool SamplesIntervalDescriptor::Interval::
        valid() const
{
    return first < last;
}

bool SamplesIntervalDescriptor::Interval::
        isConnectedTo(const Interval& r) const
{
    return last >= r.first && r.last >= first;
}

bool SamplesIntervalDescriptor::Interval::
        operator<(const Interval& r) const
{
    return first < r.first;
}

SamplesIntervalDescriptor::Interval& SamplesIntervalDescriptor::Interval::
        operator|=(const Interval& r)
{
    first = std::min(first, r.first);
    last = std::max(last, r.last);
    return *this;
}

bool SamplesIntervalDescriptor::Interval::
        operator==(const Interval& r) const
{
    return first==r.first && last==r.last;
}

SamplesIntervalDescriptor::
        SamplesIntervalDescriptor()
{
}

SamplesIntervalDescriptor::
        SamplesIntervalDescriptor(Interval r)
{
    BOOST_ASSERT( r.first < r.last );
    _intervals.push_back( r );
}

SamplesIntervalDescriptor::
        SamplesIntervalDescriptor(unsigned first, unsigned last)
{
    BOOST_ASSERT( first < last );
    Interval r = { first, last };
    _intervals.push_back( r );
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
        operator |= (const SamplesIntervalDescriptor& b)
{
    BOOST_FOREACH (const Interval& r,  b._intervals)
        operator|=( r );
    return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
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

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
        operator -= (const SamplesIntervalDescriptor& b)
{
    BOOST_FOREACH (const Interval& r,  b._intervals)
        operator-=( r );
    return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
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

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
		operator -= (const SampleType& b)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
        Interval& i = *itr;
	
		if (SampleType_MIN + b > i.first ) i.first = SampleType_MIN;
		else i.first -= b;
		if (SampleType_MIN + b > i.last ) i.last = SampleType_MIN;
		else i.last -= b;

		if ( SampleType_MIN == i.first && SampleType_MIN == i.last )
			itr = _intervals.erase( itr );
		else
			itr++;
	}

	return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
		operator += (const SampleType& b)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {
        Interval& i = *itr;
	
		if (SampleType_MAX - b < i.first ) i.first = SampleType_MAX;
		else i.first += b;
		if (SampleType_MAX - b < i.last ) i.last = SampleType_MAX;
		else i.last += b;

		if ( SampleType_MAX == i.first && SampleType_MAX == i.last )
			itr = _intervals.erase( itr );
		else
			itr++;
	}

	return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
        operator &= (const SamplesIntervalDescriptor& b)
{
	SamplesIntervalDescriptor rebuild;
	BOOST_FOREACH (const Interval& r,  b._intervals) {
		SamplesIntervalDescriptor copy = *this;
        copy&=( r );
		rebuild |= copy;
	}

	this->_intervals = rebuild._intervals;

	if (b._intervals.empty())
		_intervals.clear();

	return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
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

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
        operator*=(const float& scale)
{
    std::list<Interval>::iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        itr->first*=scale;
        itr->last*=scale;
    }

    return *this;
}

SamplesIntervalDescriptor::Interval SamplesIntervalDescriptor::
        getInterval( SampleType dt, SampleType center ) const
{
    if (0 == _intervals.size()) {
        Interval r = {SampleType_MIN, SampleType_MIN};
        return r;
    }

    std::list<Interval>::const_iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        if (itr->first >= center)
            break;
    }

    SampleType distance_to_next = SampleType_MAX;
    SampleType distance_to_prev = SampleType_MAX;

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

        SampleType start = f.first + dt*(unsigned)((center-f.first) / dt);
        Interval r = {start, std::min(start+dt, f.last) };
        return r;
    }
}

SamplesIntervalDescriptor::Interval SamplesIntervalDescriptor::
        getInterval( Interval n ) const
{
    SamplesIntervalDescriptor sid = *this & n;

    if (0 == sid._intervals.size()) {
        Interval r = {SampleType_MIN, SampleType_MIN};
        return r;
    }

    return sid.intervals().front();
}

SamplesIntervalDescriptor::Interval SamplesIntervalDescriptor::
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

void SamplesIntervalDescriptor::
        print( std::string title ) const
{
    std::stringstream ss;
    ss << *this;

    TaskTimer("%s, %s",
              title.empty()?"SamplesIntervalDescriptor":title.c_str(),
              ss.str().c_str()).suppressTiming();
}

std::ostream& operator<<( std::ostream& s, const SamplesIntervalDescriptor& i)
{
    s << "{" << i.intervals().size() << " interval" << ((i.intervals().size()==1)?"":"s");

    BOOST_FOREACH (const SamplesIntervalDescriptor::Interval& r, i.intervals())
        s << " " << r;

    return s << "}";
}

std::ostream& operator<<( std::ostream& s, const SamplesIntervalDescriptor::Interval& i)
{
    return s << "[" << i.first << ", " << i.last << "]";
}

} // namespace Signal
