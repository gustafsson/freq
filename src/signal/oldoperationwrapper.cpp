#include "oldoperationwrapper.h"
#include "buffersource.h"
#include "operation-basic.h"
#include "tfr/filter.h"
#include "tfr/transform.h"

// gpumisc
#include "Statistics.h"

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

template<typename T>
void print_buffer_stats(pBuffer b, const char* bname, const char* func, T arg, const char* file, int line) {
    TaskInfo ti(format("%s(%s): %s(%s) -> %s = %s") % file % line % func % arg % bname % b->getInterval ());

    Statistics<float>(b->getChannel (0)->waveform_data());
}

#define PRINT_BUFFER_STATS(b, arg) print_buffer_stats(b, #b, __FUNCTION__, arg, __FILE__, __LINE__)

class OldOperationTrackBufferSource: public BufferSource {
public:
    OldOperationTrackBufferSource( pBuffer waveform )
        :
          BufferSource(waveform)
    {
        //PRINT_BUFFER_STATS (waveform,"");
    }

    pBuffer read( const Interval& I ) {
        pBuffer b = BufferSource::read(I);
        //PRINT_BUFFER_STATS (b,"");

        return b;
    }
};


OldOperationWrapper::
        OldOperationWrapper(pOperation old_operation, LastRequiredInterval* required_interval)
    :
      old_operation_(old_operation),
      required_interval_(required_interval)
{
    EXCEPTION_ASSERT( old_operation );
    EXCEPTION_ASSERT( required_interval );
}


pBuffer OldOperationWrapper::
        process(pBuffer b)
{
    //PRINT_BUFFER_STATS (b,"process");

    if (!dynamic_cast<FinalSource*>(old_operation_.get ())) {
        pOperation buffer_source(new OldOperationTrackBufferSource(b));
        old_operation_->source(buffer_source);
    }

    Interval I = required_interval_->last_required_interval;
    pBuffer r = old_operation_->readFixedLength( I );

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

        OldOperationWrapper::LastRequiredInterval lri;
        OldOperationWrapper wrapper(old_operation, &lri);
        lri.last_required_interval = b->getInterval ();
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
    lri_.reset ( new OldOperationWrapper::LastRequiredInterval() );
}


Interval OldOperationDescWrapper::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    lri_->last_required_interval = I;

    if (expectedOutput)
        *expectedOutput = I;

    Interval r = I;
    pOperation s = old_operation_;

    while (s) {
        Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(s.get ());

        if (f)
            r = f->requiredInterval (r);

        s = s->DeprecatedOperation::source ();
    }

    return r;
}


OperationDesc::Ptr OldOperationDescWrapper::
        copy() const
{
    return OperationDesc::Ptr(new OldOperationDescWrapper(old_operation_));
}


Operation::Ptr OldOperationDescWrapper::
        createOperation(ComputingEngine* engine) const
{
    // This only works in one thread becuase there's only one instance of old_operation_.
    // And there's only one instance of OldOperationWrapper::LastRequiredInterval per OldOperationDescWrapper.

    if (engine)
        return Operation::Ptr();

    return Operation::Ptr(new OldOperationWrapper(old_operation_, lri_.get ()));
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
    return ("{" + old_operation_->name () + "}").c_str ();
}


Interval OldOperationDescWrapper::
        affectedInterval( const Interval& I ) const
{
    Interval r = I;
    pOperation s = old_operation_;

    while (s) {
        Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(s.get ());

        if (f)
            r = f->transform ()->transformDesc()->affectedInterval( r );

        // ignore that this is traversing the dag in the wrong direction
        s = s->DeprecatedOperation::source ();
    }

    return r;
}

} // namespace Signal



// Unit tests below


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
        pBuffer buffer(new Buffer(Interval(1,4),2,1));
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
        TargetMarker::Ptr target = write1(chain)->addTarget (target_op_wrapper);
        TargetNeeds::Ptr needs = target->target_needs();
        IInvalidator::Ptr step = write1(chain)->addOperationAt (source_op_wrapper, target);

        Signal::OperationDesc::Extent extent = read1(chain)->extent(target);
        EXCEPTION_ASSERT_EQUALS(extent.interval, buffer->getInterval ());
        write1(needs)->updateNeeds(extent.interval.get ());

        // Should wait for workers to fininsh
        EXCEPTION_ASSERT(needs->sleep(1000));

        // Should produce a cache in the target that matches the chain
        Step::Ptr target_step = read1(needs)->step ().lock();
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

        EXCEPTION_ASSERT(needs->sleep(1000));
        r = write1(target_step)->readFixedLengthFromCache(Interval(1,4));
        t[1] = 0;
        EXCEPTION_ASSERT(*r == *buffer);

        read1(chain)->rethrow_worker_exception();
    }
}

} // namespace Signal
