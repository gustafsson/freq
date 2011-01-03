#include "stftfilter.h"
#include "stft.h"

#include <stringprintf.h>
#include <CudaException.h>
#include <memory.h>

//#define TIME_StftFilter
#define TIME_StftFilter if(0)

using namespace Signal;

namespace Tfr {


StftFilter::
        StftFilter(pOperation source, pTransform t)
:   Filter(source)
{
    if (!t)
        t = Stft::SingletonP();

    BOOST_ASSERT( dynamic_cast<Stft*>(t.get()));

    transform( t );
}


Filter::ChunkAndInverse StftFilter::
        readChunk( const Signal::Interval& I )
{
    unsigned firstSample = I.first, numberOfSamples = I.count();

    TIME_StftFilter TaskTimer tt("StftFilter::readChunk ( %u, %u )", firstSample, numberOfSamples);

    Filter::ChunkAndInverse ci;

    StftFilter* f = dynamic_cast<StftFilter*>(source().get());
    if ( f && f->transform() == transform()) {
        ci = f->readChunk( I );

    } else {
        ci.inverse = _source->readFixedLength( I );

        // Compute the continous wavelet transform
        ci.chunk = (*transform())( ci.inverse );
    }

    // Apply filter
    Intervals work(ci.chunk->getInterval());
    work -= affected_samples().inverse();

    if (work)
        ci.inverse.reset();

    // Only apply filter if it would affect these samples
    if (work || !_try_shortcuts)
    {
        TIME_StftFilter Intervals(ci.chunk->getInterval()).print("StftFilter applying filter");
        (*this)( *ci.chunk );
    }

    TIME_StftFilter Intervals(ci.chunk->getInterval()).print("StftFilter after filter");

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
        throw std::invalid_argument("'transform' must be an instance of Cwt");

    // even if '_transform == t || _transform == transform()' the client
    // probably wants to reset everything when transform( t ) is called
    _invalid_samples = Intervals::Intervals_ALL;

    _transform = t;
}

} // namespace Signal
