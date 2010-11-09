#include "filter.h"
#include "signal/buffersource.h"

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
    const Signal::Intervals work(I);


    // Try to take shortcuts and avoid unnecessary work
    if (_try_shortcuts) {
        // If no samples would be non-zero, return zeros
        if (!(work - zeroed_samples()))
        {
            // Doesn't have to read from source, just create a buffer with all samples set to 0
            TIME_Filter Intervals(I).print("Filter silent");
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
                TIME_Filter Intervals(b_interval).print("Filter unaffected");
                return b;
            }

            // Explicitly return only the unaffected samples
            TIME_Filter Intervals(b_interval).print("FilterOp fixed unaffected");
            BufferSource bs(b);
            return bs.readFixedLength( (~affected & b_interval).getInterval() );
        }
    }


    // If we've reached this far, the transform will have to be computed
    ChunkAndInverse ci = readChunk( I );

    pBuffer r;
    if (ci.inverse) // TODO remove 'false &&'
        r = ci.inverse;
    else
    {
        r = _transform->inverse( ci.chunk );
    }

    TIME_Filter Intervals(ci.chunk->getInterval()).print("Filter computed chunk");
    TIME_Filter Intervals(r->getInterval()).print("Filter computed inverse");

    return r;
}

} // namespace Tfr
