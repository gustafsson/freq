#include "rectangle.h"
#include "rectangle.cu.h"

// gpumisc
#include <CudaException.h>

using namespace Tfr;

//#define TIME_FILTER
#define TIME_FILTER if(0)

namespace Filters {

Rectangle::Rectangle(float t1, float f1, float t2, float f2, bool save_inside) {
    _t1 = std::min(t1, t2);
    _f1 = std::min(f1, f2);
    _t2 = std::max(t1, t2);
    _f2 = std::max(f1, f2);
    _save_inside = save_inside;
}

void Rectangle::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt("Rectangle");

    float4 area = make_float4(
        _t1 * chunk.sample_rate - chunk.chunk_offset,
        _f1 * chunk.nScales(),
        _t2 * chunk.sample_rate - chunk.chunk_offset,
        _f2 * chunk.nScales());

    ::removeRect( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area );

    TIME_FILTER CudaException_ThreadSynchronize();
}

Signal::Intervals Rectangle::
        zeroed_samples()
{
    unsigned FS = sample_rate();
    Signal::Intervals sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::Intervals::Intervals_ALL;
        if (start_time < end_time)
            sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}

Signal::Intervals Rectangle::
        affected_samples()
{
    unsigned FS = sample_rate();
    Signal::Intervals sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::Intervals::Intervals_ALL;
        if (start_time < end_time)
            sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
}

} // namespace Filters
