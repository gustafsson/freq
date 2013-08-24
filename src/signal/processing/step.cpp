#include "step.h"

#include <boost/foreach.hpp>

#define DEBUGINFO
//#define DEBUGINFO if(0)

using namespace boost;

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
    if (deprecated == Signal::Interval::Interval_ALL)
        cache_.reset ();

    if (operation_desc_ && deprecated)
        deprecated = operation_desc_->affectedInterval(deprecated);

    DEBUGINFO TaskInfo(format("Step %1%. Deprecate %2%")
              % (operation_desc_?operation_desc_->toString ().toStdString ():"(no operation)")
              % deprecated);

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
        registerTask(Task* taskid, Signal::Interval expected_output)
{
    DEBUGINFO TaskInfo ti(format("Step %1%. Starting %2%")
              % (operation_desc_?operation_desc_->toString ().toStdString ():"(no operation)")
              % expected_output);
    running_tasks[taskid] = expected_output;
    not_started_ -= expected_output;
}


void Step::
        finishTask(Task* taskid, Signal::pBuffer result)
{
    Signal::Interval result_interval;
    if (result)
        result_interval = result->getInterval ();

    DEBUGINFO TaskInfo ti(format("Step %1%. Finish %2%")
              % (operation_desc_?operation_desc_->toString ().toStdString ():"(no operation)")
              % result_interval);

    if (result) {
        if (!cache_)
            cache_.reset(new Signal::SinkSource(result->number_of_channels ()));

        // Result must have the same number of channels and sample rate as previous cache.
        // Call deprecateCache(Signal::Interval::Interval_ALL) to erase the cache when chainging number of channels or sample rate.
        cache_->put (result);
    }

    int C = running_tasks.count (taskid);
    if (C!=1) {
        DEBUGINFO TaskInfo("C = %d, taskid = %x", C, taskid);
        EXCEPTION_ASSERTX( running_tasks.count (taskid)==1, "Could not find given task");
    }

    Signal::Intervals expected_output = running_tasks[ taskid ];

    Intervals update_miss = expected_output - result_interval;
    not_started_ |= update_miss;

    if (!expected_output) {
        DEBUGINFO TaskInfo(format("The task was not recognized. %1%") % result_interval);
    } else if (!result_interval) {
        DEBUGINFO TaskInfo(format("The task was cancelled. Restoring %1%") % update_miss);
    } else if (update_miss) {
        DEBUGINFO TaskInfo(format("These samples were supposed to be updated by the task but missed: %1%") % update_miss);
    }

    running_tasks.erase ( taskid );
    wait_for_tasks_.wakeAll ();
}


void Step::
        sleepWhileTasks(int sleep_ms)
{
    // The caller keeps a lock that is released while waiting
    gc();

    while (!running_tasks.empty ()) {
        DEBUGINFO TaskInfo(boost::format("sleepWhileTasks %d") % running_tasks.size ());
        if (!wait_for_tasks_.wait (readWriteLock(), sleep_ms < 0 ? ULONG_MAX : sleep_ms))
            return;
        gc();
    }
}


Signal::pBuffer Step::
        readFixedLengthFromCache(Signal::Interval I)
{
    return cache_ ? cache_->readFixedLength (I) : Signal::pBuffer();
}


template<typename T>
void weakmap_gc(T& m) {
    for (typename T::iterator i = m.begin (); i != m.end (); )
    {
        if (i->first.lock()) {
            i++;
        } else {
            m.erase (i);
            i = m.begin ();
        }
    }
}

void Step::
        gc()
{
    // Garbage collection, remove operation mappings whose ComputingEngine has been removed.
    weakmap_gc(operations_);

    //weakmap_gc(running_tasks);
    //wait_for_tasks_.wakeAll ();
}


} // namespace Processing
} // namespace Signal


namespace Signal {
namespace Processing {

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
