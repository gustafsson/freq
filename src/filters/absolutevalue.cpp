#include "absolutevalue.h"

using namespace Signal;

namespace Filters {


AbsoluteValue::AbsoluteValue()
    :   Operation(pOperation())
{}


pBuffer AbsoluteValue::
        read( const Interval& I )
{
    pBuffer b = source()->read (I);
    int N = b->number_of_samples ();
    for (int i=0; i<(int)b->number_of_channels (); ++i)
    {
        float* p = b->getChannel (i)->waveform_data ()->getCpuMemory ();

        for (int x=0; x<N; ++x)
        {
            if (p[x]<0)
                p[x] = -p[x];
        }
    }

    return b;
}

} // namespace Filters
