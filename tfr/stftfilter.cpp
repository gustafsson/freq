#include "tfr/stftfilter.h"
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
        t = Cwt::SingletonP();

    BOOST_ASSERT( dynamic_cast<Stft*>(t.get()));

    transform( t );
}


pChunk StftFilter::
        readChunk( const Signal::Interval& I )
{
    unsigned firstSample = I.first, numberOfSamples = I.count;

    TIME_StftFilter TaskTimer tt("StftFilter::readChunk ( %u, %u )", firstSample, numberOfSamples);

    pChunk c;

    StftFilter* f = dynamic_cast<StftFilter*>(source().get());
    if ( f && f->transform() == transform()) {
        c = f->readChunk( I );

    } else {
        pBuffer b = _source->readFixedLength( I );

        // Compute the continous wavelet transform
        c = (*transform())( b );
    }

    // Apply filter
    Intervals work(c->getInterval());
    work -= affected_samples().inverse();

    // Only apply filter if it would affect these samples
    if (work)
    {
        TIME_StftFilter Intervals(c->getInterval()).print("StftFilter applying filter");
        (*this)( *c );
    }

    TIME_StftFilter Intervals(c->getInterval()).print("StftFilter after filter");

    return c;
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
