#ifndef FANTRACKERFILTER_H
#define FANTRACKERFILTER_H

#include "tfr/cepstrumfilter.h"

namespace Tools {
namespace Support {

class FanTrackerFilter : public Tfr::CepstrumFilter
{
public:
    FanTrackerFilter();

    virtual void operator()( Tfr::Chunk& );

    virtual Signal::Intervals affected_samples();

    virtual Signal::Operation* affecting_source( const Signal::Interval& I );
    virtual void source(Signal::pOperation v);

    struct Point
    {
        float Hz;
        float amplitude;
    };

    typedef std::map<unsigned, Point> PointsT;
    std::vector<PointsT> track;

};

} // namespace Support
} // namespace Tools

#endif // FANTRACKERFILTER_H
