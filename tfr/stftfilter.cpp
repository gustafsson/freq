#include "stftfilter.h"
#include "stft.h"

#include <neat_math.h>
#include <stringprintf.h>
#include <memory.h>

#define TIME_StftFilter
//#define TIME_StftFilter if(0)

using namespace Signal;

namespace Tfr {


StftFilter::
        StftFilter(pOperation source, pTransform t)
:   Filter(source),
    exclude_end_block(false)
{
    if (!t)
        t = Stft::SingletonP();

    Stft* s = dynamic_cast<Stft*>(t.get());
    BOOST_ASSERT( s );

    s->setWindow(Stft::WindowType_Hann, 0.75f);

    _transform = t;
}


Signal::Interval StftFilter::
        requiredInterval( const Signal::Interval& I )
{
    //((Stft*)transform().get())->set_approximate_chunk_size( 1 << 12 );
    unsigned window_size = ((Stft*)transform().get())->chunk_size();
    unsigned increment   = ((Stft*)transform().get())->increment();

    // Add a margin to make sure that the STFT is computed for one block before
    // and one block after the signal. This makes it possible to do proper
    // interpolations so that there won't be any edges between blocks

    // enough for blockfilter, but not for inverse STFT
    unsigned first_chunk = 0,
             last_chunk = (I.last + window_size/2 + increment - 1)/increment;

    if (I.first >= window_size/2)
        first_chunk = (I.first - window_size/2)/increment;
    else if (last_chunk*increment < window_size + increment)
        last_chunk = (window_size + increment)/increment;


    // for inverse STFT
    first_chunk = 0;
    last_chunk = (I.last + 2*window_size - increment - 1)/increment;

    if (I.first >= window_size-increment)
        first_chunk = (I.first - (window_size-increment))/increment;
    else if (last_chunk*increment < window_size + increment)
        last_chunk = (window_size + increment)/increment;


    Interval chunk_interval (
                first_chunk*increment,
                last_chunk*increment);

    if (exclude_end_block)
    {
        if (chunk_interval.last>number_of_samples())
        {
            last_chunk = number_of_samples()/window_size;
            if (1+first_chunk<last_chunk)
                chunk_interval.last = last_chunk*window_size;
        }
    }

    return chunk_interval;
}


ChunkAndInverse StftFilter::
        computeChunk( const Signal::Interval& I )
{
    ChunkAndInverse ci;

    ci.inverse = source()->readFixedLength( requiredInterval( I ) );

    // Compute the stft transform
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
