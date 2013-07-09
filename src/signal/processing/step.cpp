#include "step.h"
#include "task.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Step::Step(Signal::OperationDesc::Ptr operation_desc, int num_channels, float /*sample_rate*/)
    :
      not_started(Signal::Intervals::Intervals_ALL),
      cache_(num_channels), // TODO add sample_rate to the cache constructor here
      operation_desc_(operation_desc)
{
}


Signal::Intervals Step::
        currently_processing() const
{
    Signal::Intervals I;

    BOOST_FOREACH(volatile Task* t, running_tasks)
    {
        I |= Task::ReadPtr( t )->expected_output();
    }

    return I;
}


Signal::Intervals Step::
        out_of_date() const
{
    return not_started | currently_processing();
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
        registerTask(volatile Task* t)
{
    running_tasks.insert (t);
}


void Step::
        finishTask(volatile Task* t, Signal::pBuffer result)
{
    cache_.put (result);
    running_tasks.erase (t);
}


Signal::pBuffer Step::
        readFixedLengthFromCache(Signal::Interval I)
{
    return cache_.readFixedLength (I);
}


float Step::
        sample_rate()
{
    return cache_.sample_rate ();
}


unsigned Step::
        num_channels()
{
    return cache_.num_channels ();
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
