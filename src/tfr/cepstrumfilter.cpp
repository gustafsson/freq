#include "cepstrumfilter.h"
#include "cepstrum.h"

#include <neat_math.h>
#include <stringprintf.h>
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
        CepstrumParams p;
        p.setWindow(StftParams::WindowType_Hann, 0.75f);
        t = pTransform(new Cepstrum(p));
    }

    Cepstrum* c = dynamic_cast<Cepstrum*>(t.get());
    BOOST_ASSERT( c );

    transform( t );
}


ChunkAndInverse CepstrumFilter::
        computeChunk( const Signal::Interval& I )
{
    TIME_CepstrumFilter TaskTimer tt("Cepstrum filter");
    Tfr::pTransform t = transform();

    unsigned chunk_size = dynamic_cast<Cepstrum*>(t.get())->params().chunk_size();
    // Add a margin to make sure that the STFT is computed for one block before
    // and one block after the signal. This makes it possible to do proper
    // interpolations so that there won't be any edges between blocks

    unsigned first_chunk = 0,
             last_chunk = (I.last + 2.5*chunk_size)/chunk_size;

    if (I.first >= 1.5*chunk_size)
        first_chunk = (I.first - 1.5*chunk_size)/chunk_size;

    ChunkAndInverse ci;

    Interval chunk_interval (
                first_chunk*chunk_size,
                last_chunk*chunk_size);
    if (exclude_end_block)
    {
        if (chunk_interval.last>number_of_samples())
        {
            last_chunk = number_of_samples()/chunk_size;
            if (1+first_chunk<last_chunk)
                chunk_interval.last = last_chunk*chunk_size;
        }
    }
    ci.inverse = source()->readFixedLength( chunk_interval );

    // Compute the cepstrum transform
    ci.chunk = (*t)( ci.inverse );

    return ci;
}


} // namespace Signal
