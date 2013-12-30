#ifndef FANTRACKERFILTER_H
#define FANTRACKERFILTER_H

#include "tfr/cepstrumfilter.h"

#include <vector>

namespace Tools {
namespace Support {

class FanTrackerFilter : public Tfr::ChunkFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    FanTrackerFilter();

    void set_number_of_channels( unsigned );
    void operator()( Tfr::ChunkAndInverse& chunk );

//    virtual Signal::Intervals affected_samples();

//    virtual Signal::DeprecatedOperation* affecting_source( const Signal::Interval& I );
//    virtual void source(Signal::pOperation v);
//    virtual void invalidate_samples(const Signal::Intervals& I);

    struct Point
    {
        float Hz;
        float amplitude;
    };

    typedef std::map<unsigned, Point> PointsT;
    std::vector<PointsT> track;
    float last_fs;

};

class FanTrackerDesc: public Tfr::CepstrumFilterDesc {
public:
    FanTrackerDesc():Tfr::CepstrumFilterDesc(Tfr::pChunkFilter(new FanTrackerFilter)){}
};

} // namespace Support
} // namespace Tools

#endif // FANTRACKERFILTER_H
