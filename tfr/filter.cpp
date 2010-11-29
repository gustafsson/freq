#include "filter.h"
#include "signal/buffersource.h"

#include <demangle.h>

//#define TIME_Filter
#define TIME_Filter if(0)

using namespace Signal;

namespace Tfr {

//////////// Filter
Filter::
        Filter( pOperation source )
            :
            Operation( source ),
            _try_shortcuts( true )
{}


Signal::pBuffer Filter::
        read(  const Signal::Interval& I )
{
    TIME_Filter TaskTimer tt("%s Filter::read( %s )", vartype(*this).c_str(),
                             I.toString().c_str());

    const Signal::Intervals work(I);


    // Try to take shortcuts and avoid unnecessary work
    if (_try_shortcuts) {
        // If no samples would be non-zero, return zeros
        if (!(work - zeroed_samples()))
        {
            // Doesn't have to read from source, just create a buffer with all samples set to 0
            TIME_Filter TaskTimer("Filter silent, %s", I.toString().c_str());
            return zeros(I);
        }

        const Signal::Intervals affected = affected_samples();
        // If no samples would be affected, return from source
        if (!(work & affected))
        {
            // Attempt a regular simple read
            pBuffer b = Signal::Operation::read( I );

            // Check if we can guarantee that everything returned from _source
            // is unaffected
            const Signal::Intervals b_interval = b->getInterval();
            if (!(affected & b_interval)) {
                TIME_Filter TaskTimer("Filter unaffected, %s", b_interval.toString().c_str());
                return b;
            }

            // Explicitly return only the unaffected samples
            TIME_Filter TaskTimer tt("FilterOp fixed unaffected, %s", b_interval.toString().c_str());
            BufferSource bs(b);
            return bs.readFixedLength( (~affected & b_interval & work).getInterval() );
        }
    }


    // If we've reached this far, the transform will have to be computed
    ChunkAndInverse ci;
    {
        TIME_Filter TaskTimer tt("%s filter computing chunk", vartype(*this).c_str());
        ci = readChunk( I );
        TIME_Filter TaskTimer("%s computed chunk %s", vartype(*this).c_str(),
                              ci.chunk->getInterval().toString().c_str()).suppressTiming();
    }

    pBuffer r;
    if (ci.inverse)
    {
        TIME_Filter TaskTimer("%s chunk is unmodified, doesn't need to compute inverse. Data = %s",
                              vartype(*this).c_str(),
                              ci.inverse->getInterval().toString().c_str()).suppressTiming();
        r = ci.inverse;
    }
    else
    {
        TIME_Filter TaskTimer tt("%s filter computing inverse", vartype(*this).c_str());
        r = _transform->inverse( ci.chunk );
        TIME_Filter TaskTimer("%s computed inverse %s", vartype(*this).c_str(), r->getInterval().toString().c_str()).suppressTiming();
    }

    return r;
}


ChunkAndInverse Filter::
        readChunk( const Signal::Interval& I )
{
    TIME_Filter TaskTimer tt("%s::readChunk [%u, %u)%u#",
                             vartype(*this).c_str(),
                             I.first, I.last, I.count());

    ChunkAndInverse ci;

    Filter* f = dynamic_cast<Filter*>(source().get());
    if ( f && f->transform() == transform()) {
        ci = f->readChunk( I );

    } else {
        ci = computeChunk( I );
    }

    // Apply filter
    Intervals work(ci.chunk->getInterval());
    work -= affected_samples().inverse();

    if (work)
        ci.inverse.reset();

    // Only apply filter if it would affect these samples
    if (work || !_try_shortcuts)
    {
        TIME_Filter TaskTimer("%s applying filter operation, %s",
                              vartype(*this).c_str(), ci.chunk->getInterval().toString().c_str());
        applyFilter( ci.chunk );
    }

    TIME_Filter TaskTimer("%s after filter operation, %s",
                          vartype(*this).c_str(), ci.chunk->getInterval().toString().c_str());

    return ci;
}


void Filter::
        applyFilter( Tfr::pChunk chunk )
{
    (*this)( *chunk );
}

} // namespace Tfr
