#include "computerms.h"
#include "signal/buffersource.h"
#include "signal/computingengine.h"

#include <boost/foreach.hpp>

using namespace Signal;

namespace Tools {
namespace Support {

ComputeRms::
        ComputeRms(shared_state<RmsValue> rms)
            :
            rms(rms)
{

}


pBuffer ComputeRms::
        process( pBuffer read_b )
{
    Intervals missing = read_b->getInterval() - rms.read ()->rms_I;

    BOOST_FOREACH(Interval i, missing)
    {
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

        {
            auto r = rms.write ();
            unsigned newL = r->rms_I.count() + L;
            r->rms = sqrt(r->rms*r->rms * r->rms_I.count()/newL + S/newL);

            r->rms_I |= b->getInterval();
        }
    }

    return read_b;
}


ComputeRmsDesc::
        ComputeRmsDesc()
    :
      rms_(new RmsValue)
{
}


Signal::Interval ComputeRmsDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Signal::Interval ComputeRmsDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    {
        auto r = rms_.write ();
        r->rms = 0;
        r->rms_I.clear ();
    }

    return Signal::Interval::Interval_ALL;
}


Signal::OperationDesc::Ptr ComputeRmsDesc::
        copy() const
{
    return Signal::OperationDesc::Ptr(new ComputeRmsDesc);
}


Signal::Operation::Ptr ComputeRmsDesc::
        createOperation(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine) || 0==engine)
        return Signal::Operation::Ptr(new ComputeRms(rms_));

    return Signal::Operation::Ptr();
}


float ComputeRmsDesc::
        rms()
{
    return rms_.read ()->rms;
}

} // namespace Support
} // namespace Tools

#include "test/randombuffer.h"

namespace Tools {
namespace Support {

void ComputeRmsDesc::
        test()
{
    // It should compute the root mean square value of a signal.
    {
        Signal::pBuffer b = Test::RandomBuffer::smallBuffer ();
        ComputeRmsDesc crd;
        Signal::Operation::Ptr o = crd.createOperation ();
        o.write ()->process (b);

        EXCEPTION_ASSERT_EQUALS(5.3572383f, crd.rms ());
    }
}

} // namespace Support
} // namespace Tools
