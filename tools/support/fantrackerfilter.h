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

};

} // namespace Support
} // namespace Tools

#endif // FANTRACKERFILTER_H
