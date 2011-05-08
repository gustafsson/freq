#include "splinefilter.h"
#include "splinefilter.cu.h"

// gpumisc
#include <CudaException.h>

// boost
#include <boost/foreach.hpp>

using namespace Tfr;

//#define TIME_SPLINEFILTER
#define TIME_SPLINEFILTER if(0)

namespace Tools { namespace Selections { namespace Support {

SplineFilter::SplineFilter(bool save_inside) {
    _save_inside = save_inside;
}


std::string SplineFilter::
        name()
{
    std::stringstream ss;
    ss << "Polygon with " << v.size() << " vertices";
    if (!this->_save_inside)
        ss << ", saving outside";
    return ss.str();
}


void SplineFilter::operator()( Chunk& chunk)
{
    TIME_SPLINEFILTER TaskTimer tt("SplineFilter chunk area (%g %g : %g %g)",
        chunk.startTime(), chunk.minHz(), chunk.endTime(), chunk.maxHz());

    unsigned N = v.size();

	std::vector<float2> p(N);

	unsigned j=0;
    float t1 = chunk.chunk_offset/chunk.sample_rate,
          t2 = (chunk.chunk_offset + chunk.nSamples())/chunk.sample_rate;

    for (unsigned i=0; i<N; ++i)
    {
        unsigned pi = (i+N-1)%N;
        unsigned ni = (i+1)%N;
        if ((v[i].t < t1 && v[ni].t < t1 && v[pi].t < t1) ||
            (v[i].t > t2 && v[ni].t > t2 && v[pi].t > t2))
		{
			continue;
		}

        p[j] = make_float2(
				v[i].t * chunk.sample_rate - chunk.chunk_offset.asFloat(),
                chunk.freqAxis.getFrequencyScalarNotClamped( v[i].f ));

        TIME_SPLINEFILTER TaskTimer("(%g %g) -> p[%u] = (%g %g)",
                  v[i].t, v[i].f, i, p[i].x, p[i].y).suppressTiming();

		j++;
    }

    if (0<j)
    {
        GpuCpuData<float2> pts(&p[0], make_uint3( j, 1, 1 ), GpuCpuVoidData::CpuMemory, true );

        ::applyspline(
                chunk.transform_data->getCudaGlobal(),
                pts.getCudaGlobal(), _save_inside );
    }

    TIME_SPLINEFILTER CudaException_ThreadSynchronize();
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

    float
        start_time = std::max(0.f, std::max(0.f, v.front().t)),
        end_time = std::max(0.f, v.front().t);

    BOOST_FOREACH( SplineVertex const& p, v )
    {
        start_time = std::min(start_time, std::max(0.f, p.t));
        end_time = std::max(end_time, p.t);
    }

    double FS = sample_rate();
    Signal::Intervals sid;
    Signal::Interval sidint(start_time*FS, end_time*FS);
    if (sidint.count())
        sid = sidint;

    return ~sid;
}

}}} // namespace Tools::Selections::Support
