#include "tfr-filter.h"
#include <functional>
#include "tfr-chunk.h"
#include "filter.cu.h"
#include <CudaException.h>
#include <float.h>
#include <boost/foreach.hpp>
#include "selection.h"

namespace Tfr {

//////////// Filter
Filter::
Filter()
:   enabled(true)
{
}

Signal::SamplesIntervalDescriptor Filter::
coveredSamples(unsigned FS)
{
    float a,b;
    this->range(a,b);
    return Signal::SamplesIntervalDescriptor((unsigned)(a*FS), (unsigned)(b*FS));
}

//////////// FilterChain

class apply_filter
{
    Chunk& t;
    bool r;
public:
    apply_filter( Chunk& t):t(t),r(true) {}

    void operator()( pFilter p) {
        if(!p.get()->enabled)
            return;

        r |= (*p)( t );
    }

    operator bool() { return r; }
};

bool FilterChain::operator()( Chunk& t) {
    return std::for_each(begin(), end(), apply_filter( t ));
}

void FilterChain::range(float& start_time, float& end_time) {
    start_time=FLT_MAX, end_time=FLT_MIN;
    float s,e;
    BOOST_FOREACH( pFilter f, *this ) {
        f->range(s, e);
        if (s<start_time) start_time=s;
        if (e>end_time) end_time=e;
    }
}

//////////// SelectionFilter

SelectionFilter::SelectionFilter( Selection s ) {
    this->s = s;
    enabled = true;
}

bool SelectionFilter::operator()( Chunk& chunk) {
    TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);

    if(dynamic_cast<RectangleSelection*>(&s))
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
    } else if(dynamic_cast<EllipseSelection*>(&s))
    {
        EllipseSelection *es = dynamic_cast<EllipseSelection*>(&s);
        float4 area = make_float4(
            es->p1.time * chunk.sample_rate - chunk.chunk_offset,
            es->p1.scale * chunk.nScales(),
            es->p2.time * chunk.sample_rate - chunk.chunk_offset,
            es->p2.scale * chunk.nScales());

        ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                      chunk.transform_data->getNumberOfElements(),
                      area );
    } else
    {
        return false;
    }

    CudaException_ThreadSynchronize();
    return true;
}

void SelectionFilter::range(float& start_time, float& end_time) {
    s.range(start_time, end_time);
}

//////////// EllipsFilter
EllipsFilter::EllipsFilter(float t1, float f1, float t2, float f2, bool save_inside)
{
    _t1 = t1;
    _f1 = f1;
    _t2 = t2;
    _f2 = f2;
    _save_inside = save_inside;
    enabled = true;
}

bool EllipsFilter::operator()( Chunk& chunk) {
    float4 area = make_float4(
            _t1 * chunk.sample_rate - chunk.chunk_offset,
            _f1 * chunk.nScales(),
            _t2 * chunk.sample_rate - chunk.chunk_offset,
            _f2 * chunk.nScales());

    TaskTimer tt(__FUNCTION__);

    ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area );

    CudaException_ThreadSynchronize();

    return true;
}

void EllipsFilter::range(float& start_time, float& end_time) {
    start_time = _t1 - fabs(_t1 - _t2);
    end_time = _t1 + fabs(_t1 - _t2);
}


//////////// SquareFilter
SquareFilter::SquareFilter(float t1, float f1, float t2, float f2, bool save_inside) {
    _t1 = std::min(t1, t2);
    _f1 = std::min(f1, f2);
    _t2 = std::max(t1, t2);
    _f2 = std::max(f1, f2);
    _save_inside = save_inside;
    enabled = true;
}

bool SquareFilter::operator()( Chunk& chunk) {
    float4 area = make_float4(
        _t1 * chunk.sample_rate - chunk.chunk_offset,
        _f1 * chunk.nScales(),
        _t2 * chunk.sample_rate - chunk.chunk_offset,
        _f2 * chunk.nScales());

    TaskTimer tt(__FUNCTION__);

    ::removeRect( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area );

    CudaException_ThreadSynchronize();

    return true;
}

void SquareFilter::range(float& start_time, float& end_time) {
    start_time = _t1;
    end_time = _t2;
}

} // namespace Tfr
