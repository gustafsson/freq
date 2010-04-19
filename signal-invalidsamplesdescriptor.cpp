#include "signal-invalidsamplesdescriptor.h"
#include <stdexcept>

namespace Signal {

bool InvalidSamplesDescriptor::Interval::
operator<(const Interval& r)
{
    return last < r.first;
}

bool InvalidSamplesDescriptor::Interval::
operator|=(const Interval& r)
{
    first = min(first, r.first);
    last = max(last, r.last);
}

InvalidSamplesDescriptor::
InvalidSamplesDescriptor()
{
}

InvalidSamplesDescriptor::
InvalidSamplesDescriptor(float first, float last)
{
    Interval r = { first, last };
    _intervals.push_back( r );
}

InvalidSamplesDescriptor& InvalidSamplesDescriptor::
operator |= (const InvalidSamplesDescriptor& b)
{
    BOOST_FOR_EACH (const Interval& r,  b._intervals)
        operator()( r );
}

InvalidSamplesDescriptor& InvalidSamplesDescriptor::
operator |= (const Interval& r)
{
    _intervals.push_back( r );
    _intervals.sort();

    for (_intervals::iterator itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        _intervals::iterator next = itr;
        next++;
        if (next==_intervals.end())
            continue;

        if ((*itr)<(*next) == (*next)<(*itr)) {
            *itr |= *next;
            itr = _intervals.remove( next );
        }
    }
}

Interval InvalidSamplesDescriptor::
popInterval( float dt, float center )
{
    if (0 == _intervals.size()) {
        Interval r = {0.f, 0.f};
        return r;
    }

    _intervals::iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); itr++) {
        if (itr->front > center)
            break;
    }
    float next=FLT_MAX;
    float prev=FLT_MAX;

    if (itr != _intervals.end()) {
        next = itr->front - center;
    }
    if (itr != _intervals.begin()) {
        _intervals::iterator itrp = itr;
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
            _intervals.remove( itr );
            return r;
        }
        Interval r = { f.first, f.first + dt };
        f.first += dt;
        return r;
    } else {
        itr--;
        Interval &f = itr;
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

        float start = f.first + dt*(unsigned)((center-f.first) / dt);
        Interval r = {start, min(start+dt, f.last) };
        if (start+dt < f.last ) {
            Interval r2 = { start+dt, f.last };
            _intervals.insert( itr, r2 );
        }
        f.last = start;
        return r;
    }
}

void InvalidSamplesDescriptor::
makeValid( Interval i )
{
    if (0 == _intervals.size()) {
        return;
    }

    _intervals::iterator itr;
    for (itr = _intervals.begin(); itr!=_intervals.end(); ) {
        if (itr->last > i.first && itr->first < i.last)
        {
            if (itr->first < i.first && itr->last > i.last) {
                Interval r2 = { i.last, itr->first };
                itr = _intervals.insert( itr, r2 );
                itr->last = i.first;
            }
            if (itr->first >= i.first)
                itr->first = i.last;
            if (itr->last <= i.last)
                itr->last = i.first;

            if (itr->last <= itr->first)
            {
                itr = _intervals.erase( itr );
                continue;
            }
        }
        itr++;
    }
}

} // namespace Signal
