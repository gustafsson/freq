#include "oldoperationwrapper.h"
#include "buffersource.h"
#include "operation-basic.h"

using namespace boost;

namespace Signal {

template<typename T>
void print_buffer(pBuffer b, const char* bname, const char* func, T arg, const char* file, int line) {
    TaskInfo ti(format("%s(%s): %s(%s) -> %s = %s") % file % line % func % arg % bname % b->getInterval ());

    float *p = b->getChannel (0)->waveform_data()->getCpuMemory ();
    for (int j=0; j<b->getInterval ().count (); ++j)
        TaskInfo("p[%d] = %g", j, p[j]);
}

#define PRINT_BUFFER(b, arg) print_buffer(b, #b, __FUNCTION__, arg, __FILE__, __LINE__)

class OldOperationTrackBufferSource: public BufferSource {
public:
    OldOperationTrackBufferSource( pBuffer waveform )
        :
          BufferSource(waveform)
    {}

    pBuffer read( const Interval& I ) {
        last_read = I;
        pBuffer b = BufferSource::read(I);

        // PRINT_BUFFER(b, I);

        return b;
    }

    Interval last_read;
};


OldOperationWrapper::
        OldOperationWrapper(pOperation old_operation)
    :
      old_operation_(old_operation)
{}


pBuffer OldOperationWrapper::
        process(pBuffer b)
{
    // PRINT_BUFFER(b, "pBuffer b");

    pOperation buffer_source(new OldOperationTrackBufferSource(b));
    if (!dynamic_cast<FinalSource*>(old_operation_.get ()))
        old_operation_->source(buffer_source);

    Interval I = b->getInterval ();

    pBuffer r = old_operation_->readFixedLength( I );

    // PRINT_BUFFER(r, old_operation_->name ());

    Interval last_read = ((OldOperationTrackBufferSource*)buffer_source.get ())->last_read;
    bool ok = (last_read & b->getInterval ()) == last_read;
    EXCEPTION_ASSERT (ok);

    return r;
}


void OldOperationWrapper::
        test()
{
    // It should use a DeprectatedOperation to compute the result of processing
    // a step.
    {
        pBuffer b(new Buffer(Interval(1,2),1,2));
        pOperation old_operation(
                    new OperationSetSilent(Signal::pOperation(), Interval(4,5)));

        OldOperationWrapper wrapper(old_operation);
        pBuffer r = wrapper.process (b);
        EXCEPTION_ASSERT( *r == *b );
    }
}


OldOperationDescWrapper::
        OldOperationDescWrapper(pOperation old_operation)
    :
      old_operation_(old_operation)
{
    EXCEPTION_ASSERT(old_operation);
}


Interval OldOperationDescWrapper::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


OperationDesc::Ptr OldOperationDescWrapper::
        copy() const
{
    return OperationDesc::Ptr(new OldOperationDescWrapper(old_operation_));
}


Operation::Ptr OldOperationDescWrapper::
        createOperation(ComputingEngine* engine) const
{
    if (engine)
        return Operation::Ptr();

    return Operation::Ptr(new OldOperationWrapper(old_operation_));
}


Signal::OperationDesc::Extent OldOperationDescWrapper::
        extent() const
{
    if (!dynamic_cast<FinalSource*>(old_operation_.get ()))
        return OperationDesc::extent ();

    Extent x;
    x.interval = old_operation_->getInterval ();
    x.number_of_channels = old_operation_->num_channels ();
    x.sample_rate = old_operation_->sample_rate ();
    return x;
}


QString OldOperationDescWrapper::
        toString() const
{
    return ("OldOperationDescWrapper {" + old_operation_->toString () + "}").c_str ();
}


Interval OldOperationDescWrapper::
        affectedInterval( const Interval& I ) const
{
    return I;
}

} // namespace Signal

#include "processing/chain.h"

using namespace boost;
using namespace Signal::Processing;

namespace Signal {

void OldOperationDescWrapper::
        test ()
{
    // It should represent an instance of DeprecatedOperation in
    // Processing::Chain.
    for (int i=0; i<10; i++) {
        // wiring
        pBuffer buffer(new Buffer(Interval(1,4),0,1));
        float *t = buffer->getChannel (0)->waveform_data()->getCpuMemory ();
        t[0] = 1;
        t[1] = 2;
        t[2] = 3;
        // PRINT_BUFFER (buffer, "");
        pOperation source_op( new BufferSource(buffer));
        //PRINT_BUFFER (source_op->read (Interval(0,2)), "");
        //PRINT_BUFFER (source_op->readFixedLength (Interval(0,2)), "");
        pOperation target_op( new OperationSetSilent(Signal::pOperation(), Signal::Interval(2,3)));

        // test that the source and target works when added to a chain
        OperationDesc::Ptr source_op_wrapper(new OldOperationDescWrapper(source_op));
        OperationDesc::Ptr target_op_wrapper(new OldOperationDescWrapper(target_op));
        Chain::Ptr chain = Chain::createDefaultChain ();
        TargetNeeds::Ptr target = write1(chain)->addTarget (target_op_wrapper);
        IInvalidator::Ptr step = write1(chain)->addOperationAt (source_op_wrapper, target);

        Signal::OperationDesc::Extent extent = read1(chain)->extent(target);
        EXCEPTION_ASSERT_EQUALS(extent.interval, buffer->getInterval ());
        write1(target)->updateNeeds(extent.interval.get ());

        // Should wait for workers to fininsh
        target->sleep();

        //read1(chain)->print_dead_workers();

        // Should produce a cache in the target that matches the chain
        Step::Ptr target_step = read1(target)->step ().lock();
        EXCEPTION_ASSERT(target_step);
        pBuffer r = write1(target_step)->readFixedLengthFromCache(Interval(1,4));
        EXCEPTION_ASSERT(r);
        EXCEPTION_ASSERT_EQUALS(r->getInterval (), buffer->getInterval ());
        EXCEPTION_ASSERT_EQUALS(r->number_of_channels (), buffer->number_of_channels ());
        t[1] = 0;
        //PRINT_BUFFER(r,"");
        //PRINT_BUFFER(buffer,"");
        EXCEPTION_ASSERT(*r == *buffer);
        t[1] = 2;

        // Modifying the source should produce a new cache in the target
        t[0] = 4;
        write1(step)->deprecateCache (Interval(1,2));

        target->sleep();
        r = write1(target_step)->readFixedLengthFromCache(Interval(1,4));
        t[1] = 0;
        EXCEPTION_ASSERT(*r == *buffer);

        read1(chain)->rethrow_worker_exception();
    }
}

} // namespace Signal
