#include "filters.h"
#include "chunk.h"
#include "filter.cu.h"
#include "sawe/selection.h"

#include <functional>
#include <CudaException.h>
#include <float.h>
#include <boost/foreach.hpp>

//#define TIME_FILTER
#define TIME_FILTER if(0)

namespace Tfr {

//////////// SelectionFilter

SelectionFilter::SelectionFilter( Selection s ) {
    this->s = s;
    enabled = true;
}

void SelectionFilter::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);

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

    TIME_FILTER CudaException_ThreadSynchronize();
    return;
}

Signal::Intervals SelectionFilter::
        ZeroedSamples( unsigned FS ) const
{
    Signal::Intervals sid;

    if (!s.inverted)
    {   float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}

Signal::Intervals SelectionFilter::
        AffectedSamples() const
{
    Signal::Intervals sid;

    if (s.inverted) {
        float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
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

void EllipsFilter::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt("EllipsFilter");

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

Signal::Intervals EllipsFilter::
        ZeroedSamples( unsigned FS ) const
{
    Signal::Intervals sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}

Signal::Intervals EllipsFilter::
        AffectedSamples() const
{
    Signal::Intervals sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
                sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
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

void SquareFilter::operator()( Chunk& chunk) {
    TIME_FILTER TaskTimer tt("SquareFilter");

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

Signal::Intervals SquareFilter::
        ZeroedSamples( unsigned FS ) const
{
    Signal::Intervals sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
            sid -= Signal::Intervals(start_time, end_time);
    }

    return sid;
}

Signal::Intervals SquareFilter::
        AffectedSamples() const
{
    Signal::Intervals sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::Intervals_ALL;
        if (start_time < end_time)
            sid -= Signal::Intervals(start_time, end_time);
    }

    return sid.inverse();
}

//////////// MoveFilter
MoveFilter::
        MoveFilter(float df)
:   _df(df)
{}

void MoveFilter::
        operator()( Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("MoveFilter");

    float df = _df * chunk.nScales();

    ::moveFilter( chunk.transform_data->getCudaGlobal(),
                  df, chunk.min_hz, chunk.max_hz, (float)chunk.sample_rate, chunk.chunk_offset );

    TIME_FILTER CudaException_ThreadSynchronize();
}


//////////// ReassignFilter
void ReassignFilter::
        operator()( Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("ReassignFilter");

    for (unsigned reassignLoop=0;reassignLoop<1;reassignLoop++)
    {
        ::reassignFilter( chunk.transform_data->getCudaGlobal(),
                      chunk.min_hz, chunk.max_hz, (float)chunk.sample_rate );
    }

    TIME_FILTER CudaException_ThreadSynchronize();
}


//////////// TonalizeFilter
TonalizeFilter::
        TonalizeFilter()
{}

void TonalizeFilter::
        operator()( Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("TonalizeFilter");

    ::tonalizeFilter( chunk.transform_data->getCudaGlobal(),
                  chunk.min_hz, chunk.max_hz, (float)chunk.sample_rate );

    TIME_FILTER CudaException_ThreadSynchronize();
}

} // namespace Tfr
