#include "computerms.h"
#include "signal/buffersource.h"

using namespace Signal;

namespace Tools {
namespace Support {

ComputeRms::
        ComputeRms(pOperation o)
            :
            Operation(o),
            rms(0)
{

}


pBuffer ComputeRms::
        read( const Interval& I )
{
    pBuffer b = Operation::read(I);
    Intervals missing = b->getInterval() - rms_I;
    if (missing.empty())
        return b;

    b = BufferSource(b).readFixedLength( missing.coveredInterval() );
    float* p = b->waveform_data()->getCpuMemory();
    unsigned L = b->number_of_samples();
    unsigned C = b->channels();
    double S = 0;
    for (unsigned i = 0; i<L*C; ++i)
    {
        S += p[i]*p[i];
    }

    S/=C;

    unsigned newL = rms_I.count() + L;
    rms = sqrt(rms*rms * rms_I.count()/newL + S/newL);

    rms_I |= b->getInterval();

    return b;
}


void ComputeRms::
        invalidate_samples(const Intervals& I)
{
    rms_I.clear();
    rms = 0;

    Operation::invalidate_samples(I);
}

} // namespace Support
} // namespace Tools
