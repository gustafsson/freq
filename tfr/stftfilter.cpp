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
//    if (!t)
//        t = Stft::SingletonP();

    if (t)
    {
        BOOST_ASSERT( dynamic_cast<Stft*>(t.get()));

        _transform = t;
    }
}


Signal::Interval StftFilter::
        requiredInterval( const Signal::Interval& I )
{
    //((Stft*)transform().get())->set_approximate_chunk_size( 1 << 12 );
    unsigned chunk_size = ((Stft*)transform().get())->chunk_size();
    // Add a margin to make sure that the STFT is computed for one block before
    // and one block after the signal. This makes it possible to do proper
    // interpolations so that there won't be any edges between blocks

    unsigned first_chunk = 0,
             last_chunk = (I.last + 2.5*chunk_size)/chunk_size;

    if (I.first >= 1.5*chunk_size)
        first_chunk = (I.first - 1.5*chunk_size)/chunk_size;

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

    return chunk_interval;
}


ChunkAndInverse StftFilter::
        computeChunk( const Signal::Interval& I )
{
    ChunkAndInverse ci;

    ci.inverse = source()->readFixedLength( requiredInterval( I ) );

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

    return ci;
}


pTransform StftFilter::
        transform() const
{
    return _transform ? _transform : Stft::SingletonP();
}


void StftFilter::
        transform( pTransform t )
{
    if (0 == dynamic_cast<Stft*>(t.get ()))
        throw std::invalid_argument("'transform' must be an instance of Tfr::Stft");

    if ( t == transform() && !_transform )
        t.reset();

    if (_transform == t )
        return;

    invalidate_samples( Signal::Interval(0, number_of_samples() ));

    _transform = t;
}

} // namespace Signal
