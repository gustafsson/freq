#include "ellipse.h"
#include "ellipse.cu.h"

// gpumisc
#include <TaskTimer.h>
#include <CudaException.h>


//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;

namespace Filters {

Ellipse::Ellipse(float t1, float f1, float t2, float f2, bool save_inside)
{
    _t1 = t1;
    _f1 = f1;
    _t2 = t2;
    _f2 = f2;
    _save_inside = save_inside;
}


void Ellipse::
        operator()( Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("Ellipse");

    float4 area = make_float4(
            _t1 * chunk.sample_rate - chunk.chunk_offset.asFloat(),
            chunk.freqAxis().getFrequencyScalar( _f1 ),
            _t2 * chunk.sample_rate - chunk.chunk_offset.asFloat(),
            chunk.freqAxis().getFrequencyScalar( _f2 ));

    ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area, _save_inside );

    TIME_FILTER CudaException_ThreadSynchronize();
}


Signal::Intervals Ellipse::
        zeroed_samples()
{
    return _save_inside ? outside_samples() : Signal::Intervals();
}


Signal::Intervals Ellipse::
        affected_samples()
{
    return (_save_inside ? Signal::Intervals() : outside_samples()).inverse();
}


Signal::Intervals Ellipse::
        outside_samples()
{
    float FS = sample_rate();

    unsigned
        start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
        end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

    Signal::Intervals sid = Signal::Intervals::Intervals_ALL;
    if (start_time < end_time)
        sid -= Signal::Intervals(start_time, end_time);

    return sid;
}

} // namespace Filters
