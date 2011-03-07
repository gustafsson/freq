#include "fantrackerfilter.h"

#include "tfr/cepstrum.h"

#include "tfr/filter.h"
#include "tools/support/fantrackerfilter.h"

using namespace Signal;

namespace Tools {
namespace Support {

    FanTrackerFilter::FanTrackerFilter()
        {
//        source_ = source;
        }

    void FanTrackerFilter::operator()( Tfr::Chunk& c )
    {
        //do stuff
    }

    Signal::Intervals FanTrackerFilter::
            affected_samples()
    {
        return Signal::Intervals(); //return empty interval
    }

} // namespace Support
} // namespace Tools
