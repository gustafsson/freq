#include "ellips.h"
#include "ellips.cu.h"

// gpumisc
#include <TaskTimer.h>
#include <CudaException.h>

//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;

namespace Filters {

Ellips::Ellips(float t1, float f1, float t2, float f2, bool save_inside)
{
    _t1 = t1;
    _f1 = f1;
    _t2 = t2;
    _f2 = f2;
    _save_inside = save_inside;
}


void Ellips::
        operator()( Chunk& chunk)
{
    TIME_FILTER TaskTimer tt("Ellips");

    float4 area = make_float4(
            _t1 * chunk.sample_rate - chunk.chunk_offset,
            _f1 * chunk.nScales(),
            _t2 * chunk.sample_rate - chunk.chunk_offset,
            _f2 * chunk.nScales());

    ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area, _save_inside );

    TIME_FILTER CudaException_ThreadSynchronize();
}


Signal::Intervals Ellips::
        zeroed_samples()
{
    unsigned FS = sample_rate();

    Signal::Intervals sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::Intervals::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}


Signal::Intervals Ellips::
        affected_samples()
{
    unsigned FS = sample_rate();

    Signal::Intervals sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::Intervals::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
}


} // namespace Filters
