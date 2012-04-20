#include "normalize.h"
#include "normalizekernel.h"
#include "cpumemorystorage.h"
#include "signal/operation-basic.h"
#include "stringprintf.h"

using namespace Signal;

namespace Filters {

Normalize::
        Normalize( unsigned normalizationRadius, pOperation source )
            :
            Operation(source),
            normalizationRadius(normalizationRadius)
{
}


std::string Normalize::
        name()
{
    return printfstring("Rolling normalization with radius %g s", normalizationRadius/sample_rate());
}


void Normalize::
        invalidate_samples(const Intervals& I)
{
    Operation::invalidate_samples(I.enlarge(normalizationRadius));
}


pBuffer Normalize::
        read( const Interval& I )
{
    Interval J = Intervals(I).enlarge(normalizationRadius).spannedInterval();
    pBuffer b;
    IntervalType silenceLength = normalizationRadius - (I.first-J.first);
    if (0 < silenceLength)
    {
        BOOST_ASSERT( 0 == J.first );

        OperationInsertSilence silence( source(), Interval(0, silenceLength) );
        J.last += silenceLength;
        b = silence.readFixedLength(J);
    }
    else
        b = source()->readFixedLength(J);

    normalizedata( b->waveform_data(), normalizationRadius );
    // normalizedata moved the data (and made the last 2*normalizationRadius elements invalid)
    b->sample_offset = J.first + normalizationRadius;

    pBuffer r( new Buffer(I.first + silenceLength, I.count(), b->sample_rate ));
    *r |= *b;
    r->sample_offset = I.first;

    return r;
}


} // namespace Filters
