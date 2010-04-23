#include "signal-samplesintervaldescriptor.h"
#include <stdexcept>
#include <boost/foreach.hpp>
#include <boost/assert.hpp>
#include <cfloat>

namespace Signal {

bool SamplesIntervalDescriptor::Interval::
operator<(const Interval& r) const
{
    return last < r.first;
}

bool SamplesIntervalDescriptor::Interval::
operator|=(const Interval& r)
{
    bool b = (*this < r) == (r < *this);
    first = std::min(first, r.first);
    last = std::max(last, r.last);
    return b;
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

    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        std::list<Interval>::iterator next = itr;
        next++;
        if (next==_intervals.end())
            continue;

        if (((*itr)<(*next)) == ((*next)<(*itr))) {
            *itr |= *next;
            _intervals.erase( next );
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

        // Check if interval 'itr' intersects with 'r'
        if (((*itr)<r) == (r<(*itr))) {

            // Check if intersection is over the start of 'itr'
            if (itr->first >= r.first && itr->last > r.last) {
                itr->first = r.last;
                itr++;
            }

            // Check if intersection is over the end of 'itr'
            else if (itr->first <= r.first && itr->last < r.last) {
                itr->last = r.first;
                itr++;
            }

            // Check if intersection is over the entire 'itr'
            else if (itr->first >= r.first && itr->last <= r.last)
                itr = _intervals.erase( itr );

            // Check if intersection is in the middle of 'itr'
            else if (itr->first < r.first && itr->last > r.last) {
                Interval j = {itr->last, r.last};
                _intervals.insert(itr, j);
                itr->last = r.first;
                itr++;
                itr++;

            // Else, error
            } else {
                throw std::logic_error("Shouldn't reach here");
            }
        } else {
            itr++;
        }
    }
    return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
operator &= (const SamplesIntervalDescriptor& b)
{
    BOOST_FOREACH (const Interval& r,  b._intervals)
        operator&=( r );
    return *this;
}

SamplesIntervalDescriptor& SamplesIntervalDescriptor::
operator &= (const Interval& r)
{
    for (std::list<Interval>::iterator itr = _intervals.begin(); itr!=_intervals.end();) {

        // Check if interval 'itr' does not intersects with 'r'
        if (((*itr)<r) != (r<(*itr))) {
            itr = _intervals.erase(itr);

        } else {

            // Check if intersection is over the start of 'itr'
            if (itr->first >= r.first && itr->last > r.last)
                itr->last = r.last;

            // Check if intersection is over the end of 'itr'
            else if (itr->first <= r.first && itr->last < r.last)
                itr->first = r.first;

            // Check if intersection is over the entire 'itr'
            else if (itr->first >= r.first && itr->last <= r.last)
            {}

            // Check if intersection is in the middle of 'itr'
            else if (itr->first < r.first && itr->last > r.last) {
                itr->first = r.first;
                itr->last = r.last;

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
popInterval( SampleType dt, SampleType center )
{
    if (0 == _intervals.size()) {
        Interval r = {0.f, 0.f};
        return r;
    }

    std::list<Interval>::iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        if (itr->first > center)
            break;
    }
    float next=FLT_MAX;
    float prev=FLT_MAX;

    if (itr != _intervals.end()) {
        next = itr->first - center;
    }
    if (itr != _intervals.begin()) {
        std::list<Interval>::iterator itrp = itr;
        itrp--;
        if (itrp->last < center )
            prev = center - itrp->last;
        else
            prev = 0;
    }
    if (next<prev) {
        Interval &f = *itr;
        if (f.last - f.first < dt ) {
            Interval r = f;
            _intervals.erase( itr );
            return r;
        }
        Interval r = { f.first, f.first + dt };
        f.first += dt;
        return r;
    } else {
        itr--;
        Interval &f = *itr;
        if (f.last - f.first < dt ) {
            Interval r = f;
            _intervals.erase( itr );
            return r;
        }

        if (f.last <= center ) {
            Interval r = { f.last-dt, f.last };
            f.last -= dt;
            return r;
        }

        SampleType start = f.first + dt*(unsigned)((center-f.first) / dt);
        Interval r = {start, std::min(start+dt, f.last) };
        if (start+dt < f.last ) {
            Interval r2 = { start+dt, f.last };
            _intervals.insert( itr, r2 );
        }
        f.last = start;
        return r;
    }
}

} // namespace Signal
