#include "selection.h"

#include <tfr/chunk.h>

// gpusmisc
#include <CudaException.h>

//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;
using namespace Signal;

namespace Filters {

Selection::Selection( SelectionParams s ) {
    this->s = s;
}

void Selection::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt(__FUNCTION__);

    throw std::logic_error("Not implemented");
    /*
    if (dynamic_cast<RectangleSelection*>(&s))
    {
        RectangleSelection *rs = dynamic_cast<RectangleSelection*>(&s);
        float4 area = make_float4(
            rs->p1.time * chunk.sample_rate - chunk.chunk_offset,
            rs->p1.scale * chunk.nScales(),
            rs->p2.time * chunk.sample_rate - chunk.chunk_offset,
            rs->p2.scale * chunk.nScales());

        ::removeRect( chunk.transform_data->getCudaGlobal().ptr(),
                      chunk.transform_data->getNumberOfElements(),
                      area );
    }
    else if (dynamic_cast<EllipseSelection*>(&s))
    {
        EllipseSelection *es = dynamic_cast<EllipseSelection*>(&s);
        float4 area = make_float4(
            es->p1.time * chunk.sample_rate - chunk.chunk_offset,
            es->p1.scale * chunk.nScales(),
            es->p2.time * chunk.sample_rate - chunk.chunk_offset,
            es->p2.scale * chunk.nScales());

        bool save_inside = false;
        ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                      chunk.transform_data->getNumberOfElements(),
                      area, save_inside );
    }
    else
    {
        return;
    }
*/
    TIME_FILTER CudaException_ThreadSynchronize();
    return;
}

Signal::Intervals Selection::
        zeroed_samples()
{
    float FS = sample_rate();

    Signal::Intervals sid;

    if (!s.inverted)
    {   float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Intervals::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}

Signal::Intervals Selection::
        affected_samples()
{
    float FS = sample_rate();

    Signal::Intervals sid;

    if (s.inverted) {
        float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Signal::Intervals::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
}

} // namespace Filters
