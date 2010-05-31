#include "tfr-filter.h"
#include <functional>
#include "tfr-chunk.h"
#include "tfr-filter.cu.h"
#include <CudaException.h>
#include <float.h>
#include <boost/foreach.hpp>
#include "selection.h"

#define TIME_FILTER
//#define TIME_FILTER if(0)

namespace Tfr {

//////////// Filter
Filter::
Filter()
:   enabled(true)
{
}

Signal::SamplesIntervalDescriptor Filter::
        getTouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
    sid -= getUntouchedSamples(FS);
    return sid;
}

//////////// FilterChain

class apply_filter
{
    Chunk& t;
public:
    apply_filter( Chunk& t):t(t) {}

    void operator()( pFilter p) {
        if(!p.get()->enabled)
            return;

        (*p)( t );
    }
};

void FilterChain::operator()( Chunk& t) {
    std::for_each(begin(), end(), apply_filter( t ));
}

Signal::SamplesIntervalDescriptor FilterChain::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    BOOST_FOREACH( pFilter f, *this ) {
        sid |= f->getZeroSamples( FS );
    }

	return sid;
}

Signal::SamplesIntervalDescriptor FilterChain::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;

    BOOST_FOREACH( pFilter f, *this ) {
        sid &= f->getUntouchedSamples( FS );
    }

    return sid;
}

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

Signal::SamplesIntervalDescriptor SelectionFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (!s.inverted)
    {   float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
                sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
}

Signal::SamplesIntervalDescriptor SelectionFilter::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (s.inverted) {
        float fstart_time, fend_time;
        s.range(fstart_time, fend_time);

        unsigned
            start_time = (unsigned)(std::max(0.f, fstart_time)*FS),
            end_time = (unsigned)(std::max(0.f, fend_time)*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
                sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
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

Signal::SamplesIntervalDescriptor EllipsFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
                sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
}

Signal::SamplesIntervalDescriptor EllipsFilter::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1 - fabsf(_t1 - _t2))*FS),
            end_time = (unsigned)(std::max(0.f, _t1 + fabsf(_t1 - _t2))*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
                sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
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

Signal::SamplesIntervalDescriptor SquareFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
            sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
}

Signal::SamplesIntervalDescriptor SquareFilter::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (!_save_inside)
    {
        unsigned
            start_time = (unsigned)(std::max(0.f, _t1)*FS),
            end_time = (unsigned)(std::max(0.f, _t2)*FS);

        sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
        if (start_time < end_time)
            sid -= Signal::SamplesIntervalDescriptor(start_time, end_time);
    }

    return sid;
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

Signal::SamplesIntervalDescriptor MoveFilter::
        getZeroSamples( unsigned /* always return constant */ ) const
{
    return Signal::SamplesIntervalDescriptor();
}

Signal::SamplesIntervalDescriptor MoveFilter::
        getUntouchedSamples( unsigned /* always return constant */ ) const
{
    return Signal::SamplesIntervalDescriptor();
}

} // namespace Tfr
