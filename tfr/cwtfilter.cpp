#include "cwtfilter.h"
#include "cwtchunk.h"
#include "cwt.h"

#include <stringprintf.h>
#include <CudaException.h>
#include <memory.h>
#include <demangle.h>

#include <boost/foreach.hpp>

//#define TIME_CwtFilter
#define TIME_CwtFilter if(0)

using namespace Signal;

namespace Tfr {


CwtFilter::
        CwtFilter(pOperation source, Tfr::pTransform t)
:   Filter(source)
{
    if (!t)
        t = Tfr::Cwt::SingletonP();

    BOOST_ASSERT( dynamic_cast<Tfr::Cwt*>(t.get()));

    transform( t );
}


ChunkAndInverse CwtFilter::
        computeChunk( const Signal::Interval& I )
{
    unsigned firstSample = I.first, numberOfSamples = I.count();

    Tfr::Cwt& cwt = *dynamic_cast<Tfr::Cwt*>(transform().get());

    unsigned c = cwt.find_bin( cwt.nScales( sample_rate() ) - 1 );
    firstSample = firstSample>>c<<c;
    numberOfSamples = (numberOfSamples + (1<<c) - 1)>>c<<c;

    unsigned time_support = cwt.wavelet_time_support_samples( sample_rate() );

    // wavelet_std_samples gets stored in cwt so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    unsigned redundant_samples = time_support;
    if (firstSample < time_support)
    {
        numberOfSamples += firstSample;
        redundant_samples = 0;
        firstSample = 0;
    }

    //unsigned first_valid_sample = firstSample;
    firstSample -= redundant_samples;

    unsigned smallest_ok_size = cwt.prev_good_size(0, sample_rate() );
    if (numberOfSamples<smallest_ok_size)
        numberOfSamples=smallest_ok_size;

    unsigned L = redundant_samples + numberOfSamples + time_support;

    TIME_CwtFilter TaskTimer tt("L=%u, redundant=%u, num=%u, support=%u, first=%u",
                 L, redundant_samples, numberOfSamples, time_support, firstSample);

    ChunkAndInverse ci;

    ci.inverse = _source->readFixedLength( Interval(firstSample,firstSample+ L) );

    TIME_CwtFilter Intervals(ci.inverse->getInterval()).print("CwtFilter readFixedLength");

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

    return ci;
}


void CwtFilter::
        applyFilter( Tfr::pChunk pchunk )
{
    TIME_CwtFilter Intervals(pchunk->getInterval()).print("CwtFilter applying filter");
    Tfr::CwtChunk* chunks = dynamic_cast<Tfr::CwtChunk*>( pchunk.get() );

    //BlockFilter* bf = dynamic_cast<BlockFilter*>(this);
    //if(bf) bf->_collection->update_sample_size( chunks );

    BOOST_FOREACH( pChunk& chunk, chunks->chunks )
    //unsigned C = chunks->chunks.size();
    //pChunk chunk = chunks->chunks[C-2];
    {
        CudaException_CHECK_ERROR();
        (*this)( *chunk );
        CudaException_ThreadSynchronize();
        CudaException_CHECK_ERROR();
    }
}


Tfr::pTransform CwtFilter::
        transform() const
{
    return _transform ? _transform : Tfr::Cwt::SingletonP();
}


void CwtFilter::
        transform( Tfr::pTransform t )
{
    if (0 == dynamic_cast<Tfr::Cwt*>(t.get ()))
        throw std::invalid_argument("'transform' must be an instance of Tfr::Cwt");

    // even if '0 == t || transform() == t' the client
    // probably wants to reset everything when transform( t ) is called
    _invalid_samples = Intervals::Intervals_ALL;

    _transform = t;
}

} // namespace Signal
