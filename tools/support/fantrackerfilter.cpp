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

        unsigned nWindows = c.nSamples();

        for (unsigned i = 0; i<nWindows; i++)
        {

        unsigned window_size = c.nScales();
        float max = 0;
        unsigned peak = -1;

        float2* p = c.transform_data->getCpuMemory() + i*window_size;

        for (unsigned m = 12 ; m < window_size/2 ; m++)
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
//        point.Hz = c.freqAxis.getFrequency(100.f);
//        point.Hz = 400;
        point.amplitude = max;

        TaskInfo("Cepstrum size: %u bins",window_size);
        TaskInfo("Cepstrum peak: %u at %g Hz",peak,point.Hz);

        unsigned pos = c.chunk_offset+i*window_size;

        if (peak == (unsigned)-1)
        {
            PointsT::iterator test = track.find(pos);
            if (test != track.end())
            {
                if (test == track.begin() || test == --track.end())
                {
                    track.erase(test);
                }
            }
        }
        else
            track[pos] = point;
        }

    }


    Signal::Intervals FanTrackerFilter::
            affected_samples()
    {
        return Signal::Intervals();
    }


    Operation* FanTrackerFilter::affecting_source( const Interval& I )
    {
        if (~zeroed_samples_recursive() & I)
            return this;

        return CepstrumFilter::affecting_source(I);
    }
} // namespace Support
} // namespace Tools
