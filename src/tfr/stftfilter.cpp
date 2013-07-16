#include "stftfilter.h"
#include "stft.h"

#include <neat_math.h>
#include <stringprintf.h>
#include <memory.h>

//#define TIME_StftFilter
#define TIME_StftFilter if(0)

using namespace Signal;

namespace Tfr {


StftFilter::
        StftFilter(pOperation source, pTransform t)
:   Filter(source),
    exclude_end_block(false)
{
    if (!t)
    {
        Tfr::StftDesc p;
        p.setWindow(Tfr::StftDesc::WindowType_Hann, 0.75f);
        t = pTransform(new Stft(p));
    }

    Stft* s = dynamic_cast<Stft*>(t.get());
    EXCEPTION_ASSERT( s );

    transform( t );
}


Signal::Interval StftFilter::
        requiredInterval( const Signal::Interval& I, Tfr::pTransform t )
{
    const StftDesc& p = ((Stft*)t.get())->desc();
    long averaging = p.averaging();
    long window_size = p.chunk_size();
    long window_increment = p.increment();
    long chunk_size  = window_size*averaging;
    long increment   = window_increment*averaging;

    // Add a margin to make sure that the inverse of the STFT will cover I
    long first_chunk = 0,
         last_chunk = (I.last + 2*window_size - window_increment-1)/increment;

    first_chunk = I.first + window_increment - window_size;
    if (first_chunk < 0)
        first_chunk = floor(first_chunk/float(increment));
    else
        first_chunk = first_chunk/increment;

    Interval chunk_interval(
                first_chunk*increment,
                last_chunk*increment);

    if (!(affected_samples() & chunk_interval))
    {
        // Add a margin to make sure that the STFT is computed for one window
        // before and one window after 'chunk_interval'.

        first_chunk = 0;
        last_chunk = (I.last + chunk_size/2 + increment - 1)/increment;

        if (I.first >= chunk_size/2)
            first_chunk = (I.first - chunk_size/2)/increment;
        else
        {
            first_chunk = floor((I.first - chunk_size/2.f)/increment);

            if (last_chunk*increment < chunk_size + increment)
                last_chunk = (chunk_size + increment)/increment;
        }

        chunk_interval = Interval(
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
    }

    return chunk_interval;
}


void StftFilter::
        invalidate_samples(const Signal::Intervals& I)
{
    // include_time_support
    const StftDesc& p = ((Stft*)transform().get())->desc();
    DeprecatedOperation::invalidate_samples( p.affectedInterval (I.spannedInterval ()) );
}


} // namespace Signal
