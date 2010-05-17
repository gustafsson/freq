#include "tfr-filter.h"
#include <functional>
#include "tfr-chunk.h"
#include "tfr-filter.cu.h"
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
        getTouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
    sid -= getUntouchedSamples(FS);
    return sid;
}

Signal::SamplesIntervalDescriptor::Interval Filter::
        coveredInterval(unsigned FS) const
{
    Signal::SamplesIntervalDescriptor sid = getTouchedSamples(FS);

    Signal::SamplesIntervalDescriptor::Interval i;
    if (sid.isEmpty()) {
        i.first = i.last = 0;
        return i;
    }

    i.first = sid.intervals().front().first;
    i.last = sid.intervals().back().last;

    return i;
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
}

Signal::SamplesIntervalDescriptor FilterChain::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);

    BOOST_FOREACH( pFilter f, *this ) {
        sid &= f->getUntouchedSamples( FS );
    }
}

//////////// SelectionFilter

SelectionFilter::SelectionFilter( Selection s ) {
    this->s = s;
    enabled = true;
}

void SelectionFilter::operator()( Chunk& chunk) {
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

        bool save_inside = false;
        ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                      chunk.transform_data->getNumberOfElements(),
                      area, save_inside );
    } else
    {
        return;
    }

    CudaException_ThreadSynchronize();
    return;
}

Signal::SamplesIntervalDescriptor SelectionFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    float start_time, end_time;
    s.range(start_time, end_time);

    sid = Signal::SamplesIntervalDescriptor(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
    sid -= Signal::SamplesIntervalDescriptor((unsigned)(start_time*FS), (unsigned)(end_time*FS));

    return sid;
}

Signal::SamplesIntervalDescriptor SelectionFilter::
        getUntouchedSamples( unsigned FS ) const
{
    return Signal::SamplesIntervalDescriptor();
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
    float4 area = make_float4(
            _t1 * chunk.sample_rate - chunk.chunk_offset,
            _f1 * chunk.nScales(),
            _t2 * chunk.sample_rate - chunk.chunk_offset,
            _f2 * chunk.nScales());

    TaskTimer tt(__FUNCTION__);

    ::removeDisc( chunk.transform_data->getCudaGlobal().ptr(),
                  chunk.transform_data->getNumberOfElements(),
                  area, _save_inside );

    CudaException_ThreadSynchronize();
}

Signal::SamplesIntervalDescriptor EllipsFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (_save_inside)
    {
        float
            start_time = _t1 - fabs(_t1 - _t2),
            end_time = _t1 + fabs(_t1 - _t2);

        sid = Signal::SamplesIntervalDescriptor(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
        sid -= Signal::SamplesIntervalDescriptor((unsigned)(start_time*FS), (unsigned)(end_time*FS));
    }

    return sid;
}

Signal::SamplesIntervalDescriptor EllipsFilter::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (!_save_inside)
    {
        float
            start_time = _t1 - fabs(_t1 - _t2),
            end_time = _t1 + fabs(_t1 - _t2);

        sid = Signal::SamplesIntervalDescriptor(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
        sid -= Signal::SamplesIntervalDescriptor((unsigned)(start_time*FS), (unsigned)(end_time*FS));
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
}

Signal::SamplesIntervalDescriptor SquareFilter::
        getZeroSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (_save_inside)
    {
        float
            start_time = _t1,
            end_time = _t2;

        sid = Signal::SamplesIntervalDescriptor(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
        sid -= Signal::SamplesIntervalDescriptor((unsigned)(start_time*FS), (unsigned)(end_time*FS));
    }

    return sid;
}

Signal::SamplesIntervalDescriptor SquareFilter::
        getUntouchedSamples( unsigned FS ) const
{
    Signal::SamplesIntervalDescriptor sid;

    if (!_save_inside)
    {
        float
            start_time = _t1,
            end_time = _t2;

        sid = Signal::SamplesIntervalDescriptor(0, Signal::SamplesIntervalDescriptor::SampleType_MAX);
        sid -= Signal::SamplesIntervalDescriptor((unsigned)(start_time*FS), (unsigned)(end_time*FS));
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
    TaskTimer tt(__FUNCTION__);

    float df = _df * chunk.nScales();

    ::moveFilter( chunk.transform_data->getCudaGlobal(),
                  df, chunk.min_hz, chunk.max_hz, chunk.sample_rate );

    CudaException_ThreadSynchronize();
}

Signal::SamplesIntervalDescriptor MoveFilter::
        getZeroSamples( unsigned FS ) const
{
    return Signal::SamplesIntervalDescriptor();
}

Signal::SamplesIntervalDescriptor MoveFilter::
        getUntouchedSamples( unsigned FS ) const
{
    return Signal::SamplesIntervalDescriptor();
}

} // namespace Tfr
