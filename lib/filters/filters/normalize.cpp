#include "normalize.h"
#include "normalizekernel.h"
#include "cpumemorystorage.h"
#include "signal/operation-basic.h"
#include "signal/computingengine.h"

using namespace Signal;

namespace Filters {

class NormalizeOperation: public Signal::Operation
{
public:
    NormalizeOperation( unsigned normalizationRadius )
        :
          normalizationRadius(normalizationRadius)
    {

    }

    Signal::pBuffer process(Signal::pBuffer b);

private:
    unsigned normalizationRadius;
};


Signal::pBuffer NormalizeOperation::
        process(Signal::pBuffer b)
{
    Interval J = b->getInterval ();
    Interval I = Signal::Intervals(J).shrink (normalizationRadius).spannedInterval ();

    for (unsigned c=0; c<b->number_of_channels (); ++c)
       normalizedata( b->getChannel (c)->waveform_data(), normalizationRadius );

    // normalizedata moved the data (and made the last 2*normalizationRadius elements invalid)
    Signal::IntervalType sample_offset = J.first + normalizationRadius;

    pBuffer r( new Buffer(I.first, I.count(), b->sample_rate(), b->number_of_channels ()));
    for (unsigned c=0; c<b->number_of_channels (); ++c)
    {
        MonoBuffer mb(sample_offset, b->getChannel (c)->waveform_data (), b->sample_rate ());
        *r->getChannel (c) |= mb;
    }

    return r;
}


Normalize::
        Normalize( unsigned normalizationRadius, Method method )
            :
            normalizationRadius(normalizationRadius)
{
    EXCEPTION_ASSERTX (method == Method_InfNorm, "Only Method_InfNorm is supported");
}


Normalize::
        Normalize()
{}


Signal::Interval Normalize::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;

    return Signal::Intervals(I).enlarge(normalizationRadius).spannedInterval ();
}


Signal::Interval Normalize::
        affectedInterval( const Signal::Interval& I ) const
{
    return Signal::Intervals(I).enlarge(normalizationRadius).spannedInterval ();
}


Signal::OperationDesc::ptr Normalize::
        copy() const
{
    return Signal::OperationDesc::ptr(new Normalize(normalizationRadius));
}


Signal::Operation::ptr Normalize::
        createOperation(Signal::ComputingEngine* engine) const
{
    if (engine == 0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Signal::Operation::ptr(new NormalizeOperation(normalizationRadius));

    return Signal::Operation::ptr();
}


QString Normalize::
        toString() const
{
    return (boost::format("Rolling normalization with radius %g samples") % normalizationRadius).str().c_str();
}


unsigned Normalize::
        radius()
{
    return normalizationRadius;
}


} // namespace Filters

#include "test/randombuffer.h"
#include "test/operationmockups.h"
#include "signal/buffersource.h"

namespace Filters {

void Normalize::
        test ()
{
    // It should normalize the signal strength.
    {
        Normalize n(10);
        Signal::pBuffer b = Test::RandomBuffer::smallBuffer ();
        Signal::Interval expectedOutput;
        Signal::Interval r = n.requiredInterval (b->getInterval (), &expectedOutput);
        EXCEPTION_ASSERT_EQUALS(b->getInterval (), expectedOutput);

        Signal::pBuffer b2 = Signal::BufferSource(b).readFixedLength (r);

        Signal::Operation::ptr o = n.createOperation (0);
        Signal::pBuffer b3 = o->process(b2);
        EXCEPTION_ASSERT_EQUALS(expectedOutput, b3->getInterval ());

        for (unsigned c=0; c<b->number_of_channels (); c++)
        {
            float* p1 = b->getChannel (c)->waveform_data ()->getCpuMemory ();
            float* p2 = b3->getChannel (c)->waveform_data ()->getCpuMemory ();
            float f = p1[0] / p2[0];
            float normsum = 0;

            for (int i=0; i<b->number_of_samples (); i++)
            {
                normsum += std::fabs(p2[i]);
                float d = std::fabs(p1[i] - p2[i]*f);
                EXCEPTION_ASSERT_LESS(d, 1e-5);
            }

            double N = n.radius () + 1 + n.radius ();
            normsum /= N;
            EXCEPTION_ASSERT_LESS(std::fabs(normsum-1),1e-5);
        }
    }
}

} // namespace Filters
