#ifndef SPLINEFILTER_H
#define SPLINEFILTER_H

#include "tfr/cwtfilter.h"
#include <vector>

namespace Tools { namespace Selections { namespace Support {

class SplineFilter: public Tfr::CwtFilter
{
public:
    SplineFilter(bool save_inside=true);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

	struct SplineVertex
    {
        float t, f; // t in seconds, f in hertz (as opposed to scale in Heightmap::Position)
    };

    std::vector<SplineVertex> v;
    bool _save_inside;

private:
    Signal::Intervals outside_samples();
};

}}} // namespace Tools::Selections::Support

#endif // SPLINEFILTER_H
