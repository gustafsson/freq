#ifndef TFRFILTER_H
#define TFRFILTER_H

#include "selection.h"
#include "signal-samplesintervaldescriptor.h"

#include <list>
#include <boost/shared_ptr.hpp>


namespace Tfr {
struct Chunk;

class Filter
{
public:
    Filter();
    virtual ~Filter() {}

    virtual void operator()( Chunk& ) = 0;
    virtual void range(float& start_time, float& end_time) = 0;

    virtual Signal::SamplesIntervalDescriptor coveredSamples( unsigned FS );

    bool enabled;
};
typedef boost::shared_ptr<Filter> pFilter;

class FilterChain: public Filter, public std::list<pFilter>
{
public:
    virtual void operator()( Chunk& );
    virtual void range(float& start_time, float& end_time);
};

class SelectionFilter: public Filter
{
public:
    SelectionFilter( Selection s );

    virtual void operator()( Chunk& );
    virtual void range(float& start_time, float& end_time);

    Selection s;

private:
    SelectionFilter& operator=(const SelectionFilter& );
    SelectionFilter(const SelectionFilter& );
};

class EllipsFilter: public Filter
{
public:
    EllipsFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Chunk& );
    virtual void range(float& start_time, float& end_time);

    float _t1, _f1, _t2, _f2;

private:
    bool _save_inside;
};

class SquareFilter: public Filter
{
public:
    SquareFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Chunk& );
    virtual void range(float& start_time, float& end_time);

    float _t1, _f1, _t2, _f2;

private:
    bool _save_inside;
};

class MoveFilter: public Filter
{
public:
    MoveFilter(float df);

    virtual void operator()( Chunk& );
    virtual void range(float& start_time, float& end_time);

    float _df;
};

} // namespace Tfr

#endif // TFRFILTER_H
