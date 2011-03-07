#include "fantrackerfilter.h"

#include "tfr/cepstrum.h"

#include "tfr/filter.h"
#include "tools/support/fantrackerfilter.h"

using namespace Signal;

namespace Tools {
namespace Support {

    FanTrackerFilter::FanTrackerFilter()
        {
            _try_shortcuts = false; // johans hack to NOT skip calculation
        }

    void FanTrackerFilter::operator()( Tfr::Chunk& c )
    {
        //find the peak, store time, freq and amp in the map called track.
        float2* p = c.transform_data->getCpuMemory();

        unsigned window_size = c.nScales();
        float max = 0;
        unsigned peak = 0;

        for (unsigned m = 0 ; m < window_size/100 ; m++)
        {
            float2 & v = p[m];
            if( v.x*v.x + v.y*v.y > max)
            {
                max = v.x*v.x + v.y*v.y;
                peak = m;
            }
        }

        Point point;
        point.Hz = c.freqAxis.getFrequency(peak);
        point.amplitude = max;
        track[(c.chunk_offset)] = point;

    }

    Signal::Intervals FanTrackerFilter::
            affected_samples()
    {
        return Signal::Intervals(); //return empty interval. this seems to not work correctly
    }

} // namespace Support
} // namespace Tools
