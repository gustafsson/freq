#include "step.h"
#include "task.h"

namespace Signal {
namespace Processing {

Step::Step(Signal::OperationDesc::Ptr operation_desc, int num_channels, float /*sample_rate*/)
    :
      cache(num_channels), // TODO add sample_rate to the cache constructor here
      operation_desc_(operation_desc)
{
}


Signal::Intervals Step::
        currently_processing() const
{
    Signal::Intervals I;

    for (size_t i=0; i<running_tasks.size (); i++) {
        I |= Task::ReadPtr( running_tasks[i] )->expected_output();
    }

    return I;
}


Signal::Intervals Step::
        out_of_date() const
{
    return todo | currently_processing();
}


Signal::Operation::Ptr Step::
        operation(Signal::ComputingEngine::Ptr ce)
{
    gc();

    Signal::ComputingEngine::WeakPtr wp(ce);
    OperationMap::iterator oi = operations_.find (wp);

    if (oi != operations_.end ())
    {
        return oi->second;
    }

    Signal::Operation::Ptr o = operation_desc_->createOperation (ce.get ());
    operations_[wp] = o;

    return o;
}


Signal::OperationDesc::Ptr Step::
        operation_desc () const
{
    return operation_desc_;
}


void Step::
        operation_desc (Signal::OperationDesc::Ptr p)
{
    operation_desc_ = p;
}


void Step::
        gc()
{
    // Garbage collection, remove operation mappings whose ComputingEngine has been removed.

    for (OperationMap::iterator omi = operations_.begin ();
         omi != operations_.end (); omi++)
    {
        Signal::ComputingEngine::Ptr p(omi->first);
        if (0 == p.get ()) {
            operations_.erase (omi);
            omi = operations_.begin ();
        }
    }
}


void Step::
        test()
{
    // It should store stuff in the cache
    {
        Signal::pBuffer b(new Buffer(Interval(60,70), 40, 7));
        for (unsigned c=0; c<b->number_of_channels (); ++c)
        {
            float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
            for (int i=0; i<b->number_of_samples (); ++i)
                p[i] = c + i/(float)b->number_of_samples ();
        }
        Signal::OperationDesc::Ptr bs(new Signal::BufferSource(b));

        Step s(bs, b->number_of_samples (), b->sample_rate ());
        //s.cache.put ();
    }
/*    Operation::Ptr o = s.createOperation (0);
    Operation::test (o, &s);

    BufferSource
    Signal::OperationDesc
    Step s()*/
}


} // namespace Processing
} // namespace Signal
