#include "fantrackerfilter.h"

#include "tfr/cepstrum.h"
#include "tfr/filter.h"
#include "tfr/chunk.h"
#include "tools/support/fantrackerfilter.h"

#include <boost/foreach.hpp>

using namespace Signal;

namespace Tools {
namespace Support {

    FanTrackerFilter::FanTrackerFilter()
    {
    }


    void FanTrackerFilter::operator()( Tfr::Chunk& c )
    {
        BOOST_ASSERT( this->track.size() == num_channels() );

        //find the peak, store time, freq and amp in the map called track.

        unsigned nWindows = c.nSamples();

        for (unsigned i = 0; i<nWindows; i++)
        {

        unsigned window_size = c.nScales();
        float max = 0;
        unsigned peak = -1;

        Tfr::ChunkElement* p = c.transform_data->getCpuMemory() + i*window_size;

        //unsigned start = c.freqAxis.getFrequencyScalar( 100 );
        //unsigned stop = c.freqAxis.getFrequencyScalar( 50 );

        for (unsigned m = 20 ; m < window_size/2 ; m++)
        {
            Tfr::ChunkElement & v = p[m];
            float A = norm(v);
            if( A > max)
            {
                max = A;
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

        unsigned pos = c.chunk_offset.asInteger() + i*window_size;

        PointsT& track = this->track[ get_channel() ];

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

    void FanTrackerFilter::invalidate_samples(Signal::Intervals const & intervals)
    {
        for (unsigned i = 0; i < this->track.size(); i++)
        {
            PointsT& track = this->track[ i ];

            for (PointsT::iterator ii = track.begin(); ii != track.end();)
            {
                if (intervals.testSample( ii->first )) track.erase(ii++);
                else ii++;
            }
        }
        CepstrumFilter::invalidate_samples(intervals);
    }

    Operation* FanTrackerFilter::affecting_source( const Interval& I )
    {
        if (~zeroed_samples_recursive() & I)
            return this;

        return CepstrumFilter::affecting_source(I);
    }


    void FanTrackerFilter::
            source(pOperation v)
    {
        if (v)
            track.resize( v->num_channels() );

        return CepstrumFilter::source(v);
    }
} // namespace Support
} // namespace Tools
