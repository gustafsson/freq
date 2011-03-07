#ifndef FANTRACKERFILTER_H
#define FANTRACKERFILTER_H

#include "tfr/cepstrumfilter.h"

namespace Tools {
namespace Support {

class FanTrackerFilter : public Tfr::CepstrumFilter
{
public:
    FanTrackerFilter( Signal::pOperation source=Signal::pOperation(),
                       Tfr::pTransform transform=Tfr::pTransform() );

    virtual void operator()( Tfr::Chunk& );

};

} // namespace Support
} // namespace Tools

#endif // FANTRACKERFILTER_H
