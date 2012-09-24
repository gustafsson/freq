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
    const pBuffer read_b = Operation::read(I);
    Intervals missing = read_b->getInterval() - rms_I;

    while (missing)
    {
        Interval i = missing.fetchFirstInterval();
        missing -= i;
        pBuffer b = BufferSource(read_b).readFixedLength( i );
        unsigned L = b->number_of_samples();
        unsigned C = b->number_of_channels ();
        double S = 0;
        for (unsigned c = 0; c<C; ++c)
        {
            float* p = b->getChannel (c)->waveform_data()->getCpuMemory();
            for (unsigned i = 0; i<L; ++i)
            {
                S += p[i]*p[i];
            }
        }

        S/=C;

        unsigned newL = rms_I.count() + L;
        rms = sqrt(rms*rms * rms_I.count()/newL + S/newL);

        rms_I |= b->getInterval();
    }

    return read_b;
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
