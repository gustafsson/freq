#include "absolutevalue.h"

using namespace Signal;

namespace Filters {


pBuffer AbsoluteValue::
        process( pBuffer b )
{
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


Signal::Interval AbsoluteValueDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval AbsoluteValueDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return I;
}


Signal::OperationDesc::ptr AbsoluteValueDesc::
        copy() const
{
    return Signal::OperationDesc::ptr(new AbsoluteValueDesc());
}


Signal::Operation::ptr AbsoluteValueDesc::
        createOperation(Signal::ComputingEngine* ) const
{
    return Signal::Operation::ptr(new AbsoluteValue);
}

} // namespace Filters

#include "test/randombuffer.h"

namespace Filters {

void AbsoluteValueDesc::
        test()
{
    // It should compute the absolute value of a signal.
    {
        AbsoluteValueDesc avd;
        Signal::Operation::ptr o = avd.createOperation (0);
        Signal::pBuffer b = o->process (Test::RandomBuffer::smallBuffer ());
        float b1[] = {3, 7, 8, 5, 3};
        float b2[] = {1, 9, 3, 6, 2};
        EXCEPTION_ASSERT(0 == memcmp(b1, b->getChannel (0)->waveform_data ()->getCpuMemory (), sizeof(b1)));
        EXCEPTION_ASSERT(0 == memcmp(b2, b->getChannel (1)->waveform_data ()->getCpuMemory (), sizeof(b2)));
    }
}

} // namespace Filters
