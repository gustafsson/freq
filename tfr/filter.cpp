#include "filter.h"
#include "signal/buffersource.h"
#include "tfr/chunk.h"
#include "tfr/transform.h"

#include <demangle.h>

#define TIME_Filter
//#define TIME_Filter if(0)

#define TIME_FilterReturn
//#define TIME_FilterReturn if(0)

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
        if (!(work - zeroed_samples_recursive()))
        {
            // Doesn't have to read from source, just create a buffer with all samples set to 0
            TIME_Filter TaskTimer("Filter silent, %s", I.toString().c_str());
            return zeros(I);
        }

        const Signal::Intervals affected = affected_samples();
        // If no samples would be affected, return from source
        if (this!=affecting_source(I) && !(work & affected))
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
            return bs.readFixedLength( (~affected & b_interval & work).fetchFirstInterval() );
        }
    }


    // If we've reached this far, the transform will have to be computed
    ChunkAndInverse ci;
    {
        TIME_Filter TaskTimer tt("%s filter computing chunk", vartype(*this).c_str());
        ci = readChunk( I );
        TIME_FilterReturn TaskInfo("%s filter computed chunk %s", vartype(*this).c_str(),
                              ci.chunk->getInterval().toString().c_str());
    }

    pBuffer r;
    if (ci.inverse)
    {
        TIME_Filter TaskInfo("%s filter chunk is unmodified, doesn't need to compute inverse. Data = %s",
                              vartype(*this).c_str(),
                              ci.inverse->getInterval().toString().c_str());
        r = ci.inverse;
    }
    else
    {
        TIME_Filter TaskTimer tt("%s filter computing inverse", vartype(*this).c_str());
        r = transform()->inverse( ci.chunk );
        TIME_FilterReturn TaskInfo("%s filter computed inverse %s", vartype(*this).c_str(), r->getInterval().toString().c_str());
    }

    return r;
}


Operation* Filter::
        affecting_source( const Interval& I )
{
    if (!_try_shortcuts)
        return this;
    return Operation::affecting_source( I );
}


unsigned Filter::
        prev_good_size( unsigned current_valid_samples_per_chunk )
{
    return transform()->prev_good_size( current_valid_samples_per_chunk, sample_rate() );
}


unsigned Filter::
        next_good_size( unsigned current_valid_samples_per_chunk )
{
    return transform()->next_good_size( current_valid_samples_per_chunk, sample_rate() );
}


ChunkAndInverse Filter::
        readChunk( const Signal::Interval& I )
{
    TIME_Filter TaskTimer tt("%s Filter::readChunk %s",
                             vartype(*this).c_str(),
                             I.toString().c_str());

    ChunkAndInverse ci;

    Filter* f = dynamic_cast<Filter*>(source()->affecting_source(I));
    if ( false && f && f->transform() == transform()) {
        ci = f->readChunk( I );

    } else {
        TIME_Filter
        {
            if (f)
            {
                TaskInfo("Filter affecting source is %s, and is not using the same transform()", vartype(*f).c_str());
            }
            else
            {
                Operation* o = source()->affecting_source(I);
                if (o != source().get())
                    TaskInfo("source()->affecting_source(I) is %s", vartype(*o).c_str());
            }
            TaskInfo("source() is %s", vartype(*source().get()).c_str());
        }

        TIME_Filter TaskTimer tt("Calling %s::computeChunk",
                                 vartype(*this).c_str());

        ci = computeChunk( I );

#ifdef _DEBUG
        //Signal::Interval cii = ci.chunk->getInterval();
        Signal::Interval cii = ci.chunk->getCoveredInterval();
        BOOST_ASSERT( cii & I );
#endif
    }

    // Apply filter
    Intervals work(ci.chunk->getInterval());
    work -= affected_samples().inverse();

    if (work)
        ci.inverse.reset();

    // Only apply filter if it would affect these samples
    if (this==affecting_source(I) || work || !_try_shortcuts)
    {
        TIME_Filter TaskTimer tt("%s filter applying operation, %s",
                              vartype(*this).c_str(), ci.chunk->getInterval().toString().c_str());
        applyFilter( ci );
        TIME_FilterReturn TaskInfo("%s filter after operation",
                              vartype(*this).c_str());
    }

    return ci;
}


void Filter::
        applyFilter( ChunkAndInverse& chunk )
{
    (*this)( *chunk.chunk );
}

} // namespace Tfr
