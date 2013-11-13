#include "cepstrumfilter.h"
#include "cepstrum.h"

#include "neat_math.h"
#include "stringprintf.h"
#include <memory.h>

#define TIME_CepstrumFilter
//#define TIME_CepstrumFilter if(0)

using namespace Signal;

namespace Tfr {


CepstrumFilter::
        CepstrumFilter(pOperation source, pTransform t)
:   Filter(source),
    exclude_end_block(false)
{
    if (!t)
    {
        CepstrumDesc p;
        p.setWindow(StftDesc::WindowType_Hann, 0.75f);
        t = pTransform(new Cepstrum(p));
    }

    Cepstrum* c = dynamic_cast<Cepstrum*>(t.get());
    EXCEPTION_ASSERT( c );

    transform( t );
}


Signal::Interval CepstrumFilter::
        requiredInterval( const Signal::Interval& I, Tfr::pTransform t )
{
    const CepstrumDesc& p = ((Cepstrum*)t.get())->desc();
    long averaging = p.averaging();
    long window_size = p.chunk_size();
    long window_increment = p.increment();
    long chunk_size  = window_size*averaging;
    long increment   = window_increment*averaging;

    // Add a margin to make sure that the STFT is computed for one window
    // before and one window after 'chunk_interval'.

    long first_chunk = 0;
    long last_chunk = (I.last + chunk_size/2 + increment - 1)/increment;

    if (I.first >= chunk_size/2)
        first_chunk = (I.first - chunk_size/2)/increment;
    else
    {
        first_chunk = floor((I.first - chunk_size/2.f)/increment);

        if (last_chunk*increment < chunk_size + increment)
            last_chunk = (chunk_size + increment)/increment;
    }

    Interval chunk_interval(
                first_chunk*increment,
                last_chunk*increment);

    if (exclude_end_block)
    {
        if (chunk_interval.last>number_of_samples())
        {
            last_chunk = number_of_samples()/chunk_size;
            if (1+first_chunk<last_chunk)
                chunk_interval.last = last_chunk*chunk_size;
        }
    }

    return chunk_interval;
}


void CepstrumFilter::
        invalidate_samples(const Signal::Intervals& I)
{
    const CepstrumDesc& p = ((Cepstrum*)transform().get())->desc();
    int window_size = p.chunk_size();
    int increment   = p.increment();

    // include_time_support
    Signal::Intervals J = I.enlarge(window_size-increment);
    DeprecatedOperation::invalidate_samples( J );
}


} // namespace Signal
