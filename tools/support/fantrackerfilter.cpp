#include "fantrackerfilter.h"

#include "tfr/cepstrum.h"

#include "tfr/filter.h"
#include "tools/support/fantrackerfilter.h"

using namespace Signal;

namespace Tools {
namespace Support {

    FanTrackerFilter::FanTrackerFilter(pOperation source, Tfr::pTransform t)
        {
//        source_ = source;
//        t_ = t;
        }

    void FanTrackerFilter::operator()( Tfr::Chunk& c )
    {
        //do stuff
    }

} // namespace Support
} // namespace Tools
