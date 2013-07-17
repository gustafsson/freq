#include "step.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Step::Step(Signal::OperationDesc::Ptr operation_desc)
    :
        not_started_(Signal::Intervals::Intervals_ALL),
        operation_desc_(operation_desc)
{
}


Signal::Intervals Step::
        currently_processing() const
{
    Signal::Intervals I;

    BOOST_FOREACH(RunningTaskMap::value_type ti, running_tasks)
    {
        I |= ti.second;
    }

    return I;
}


Signal::Intervals Step::
        deprecateCache(Signal::Intervals deprecated)
{
    if (operation_desc_)
        deprecated = operation_desc_->affectedInterval(deprecated);

    not_started_ |= deprecated;
    return deprecated;
}


Signal::Intervals Step::
        not_started() const
{
    return not_started_;
}


Signal::Intervals Step::
        out_of_date() const
{
    return not_started_ | currently_processing();
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
        registerTask(volatile Task* t, Signal::Interval expected_output)
{
    running_tasks[t] = expected_output;
    not_started_ -= expected_output;
}


void Step::
        finishTask(volatile Task* t, Signal::pBuffer result)
{
    if (!cache_) cache_.reset(new Signal::SinkSource(result->number_of_channels ()));
    cache_->put (result);
    running_tasks.erase (t);

    wait_for_tasks_.wakeAll ();
}


void Step::
        sleepWhileTasks()
{
    // The caller keeps a lock that is released while waiting

    while (!running_tasks.empty ()) {
        wait_for_tasks_.wait (readWriteLock());
    }
}


Signal::pBuffer Step::
        readFixedLengthFromCache(Signal::Interval I)
{
    return cache_ ? cache_->readFixedLength (I) : Signal::pBuffer();
}


void Step::
        gc()
{
    // Garbage collection, remove operation mappings whose ComputingEngine has been removed.

    for (OperationMap::iterator omi = operations_.begin ();
         omi != operations_.end (); )
    {
        Signal::ComputingEngine::Ptr p = omi->first.lock();
        if (!p) {
            operations_.erase (omi);
            omi = operations_.begin ();
        } else
            omi++;
    }
}


void Step::
        test()
{
    // It should keep a cache (for an OpertionDesc) and keep track of things to work on.
    {
        // Create an OperationDesc
        Signal::pBuffer b(new Buffer(Interval(60,70), 40, 7));
        for (unsigned c=0; c<b->number_of_channels (); ++c)
        {
            float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
            for (int i=0; i<b->number_of_samples (); ++i)
                p[i] = c + 1+i/(float)b->number_of_samples ();
        }

        // Create a Step
        Step s((Signal::OperationDesc::Ptr()));

        // It should contain information about what's out_of_date and what's currently being updated.
        s.registerTask(0, b->getInterval ());
        EXCEPTION_ASSERT_EQUALS(s.not_started (), ~Signal::Intervals(b->getInterval ()));
        EXCEPTION_ASSERT_EQUALS(s.out_of_date(), Signal::Intervals::Intervals_ALL);
        s.finishTask(0, b);
        EXCEPTION_ASSERT_EQUALS(s.out_of_date(), ~Signal::Intervals(b->getInterval ()));

        EXCEPTION_ASSERT( *b == *s.readFixedLengthFromCache (b->getInterval ()) );
    }


}


} // namespace Processing
} // namespace Signal
