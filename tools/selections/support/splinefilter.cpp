#include "splinefilter.h"
#include "splinefilter.cu.h"

#include <boost/foreach.hpp>

// gpumisc
#include <CudaException.h>

using namespace Tfr;

#define TIME_FILTER
//#define TIME_FILTER if(0)

namespace Tools { namespace Selections { namespace Support {

SplineFilter::SplineFilter(bool save_inside) {
    _save_inside = save_inside;
}


void SplineFilter::operator()( Chunk& chunk)
{
    TIME_FILTER TaskTimer tt("SplineFilter chunk area (%g %g : %g %g)",
        chunk.startTime(), chunk.min_hz, chunk.endTime(), chunk.max_hz);

    unsigned N = v.size();
    GpuCpuData<float2> pts(0, make_uint3( N, 1, 1 ) );
    float2 *p = pts.getCpuMemory();

    for (unsigned i=0; i<N; ++i)
    {
        p[i] = make_float2(
                v[i].t * chunk.sample_rate - chunk.chunk_offset.asFloat(),
                chunk.freqAxis().getFrequencyScalar( v[i].f ));

        TIME_FILTER TaskTimer("(%g %g) -> p[%u] = (%g %g)",
                  v[i].t, v[i].f, i, p[i].x, p[i].y).suppressTiming();
    }

    ::applyspline(
            chunk.transform_data->getCudaGlobal(),
            pts.getCudaGlobal(), _save_inside );

    TIME_FILTER CudaException_ThreadSynchronize();
}


Signal::Intervals SplineFilter::
        zeroed_samples()
{
    return _save_inside ? outside_samples() : Signal::Intervals();
}


Signal::Intervals SplineFilter::
        affected_samples()
{
    return (_save_inside ? Signal::Intervals() : outside_samples()).inverse();
}


Signal::Intervals SplineFilter::
        outside_samples()
{
    if (v.size() < 2)
        return Signal::Intervals::Intervals_ALL;

    float FS = sample_rate();

    unsigned
        start_time = (unsigned)(std::max(0.f, v.front().t)*FS),
        end_time = (unsigned)(std::max(0.f, v.front().t)*FS);

    BOOST_FOREACH( SplineVertex const& p, v )
    {
        start_time = std::min(start_time, std::max((unsigned)0, (unsigned)(p.t*FS)));
        end_time = std::max(end_time, (unsigned)(p.t*FS));
    }

    Signal::Intervals sid = Signal::Intervals::Intervals_ALL;
    if (start_time < end_time)
        sid -= Signal::Intervals(start_time, end_time);

    return sid;
}

}}} // namespace Tools::Selections::Support
