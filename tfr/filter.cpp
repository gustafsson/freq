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
            enabled(true)
{}


Signal::pBuffer Filter::
        read(  const Signal::Interval& I )
{
    const Signal::Intervals work(I);


    // Try to take shortcuts and avoid unnecessary work
    {
        // If no samples would be non-zero, return zeros
        if (!(work - zeroed_samples()))
        {
            // Doesn't have to read from source, just create a buffer with all samples set to 0
            pBuffer b( new Buffer( I.first, I.count, sample_rate() ));

            ::memset( b->waveform_data->getCpuMemory(), 0, b->waveform_data->getSizeInBytes1D());

            TIME_Filter Intervals(b->getInterval()).print("Filter silent");
            return b;
        }

        // If no samples would be affected, return from source
        if (!(work & affected_samples()))
        {
            // Attempt a regular simple read
            pBuffer b = _source->read( I );

            // Check if we can guarantee that everything returned from _source
            // is unaffected
            if (!(affected_samples() & b->getInterval())) {
                TIME_Filter Intervals(b->getInterval()).print("Filter unaffected");
                return b;
            }

            // Explicitly return only the unaffected samples
            TIME_Filter Intervals(b->getInterval()).print("FilterOp fixed unaffected");
            BufferSource bs(b);
            return bs.readFixedLength( (~affected_samples() & b->getInterval()).getInterval() );
        }
    }


    // If we've reached this far, the transform will have to be computed
    pChunk c = readChunk( I );

    Signal::pBuffer r = transform()->inverse( c );

    _invalid_samples -= r->getInterval();
    TIME_Filter Intervals(c->getInterval()).print("Filter after inverse");
    return r;
}

} // namespace Tfr
