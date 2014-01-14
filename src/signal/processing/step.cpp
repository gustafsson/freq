#include "step.h"
#include "test/operationmockups.h"

#include "TaskTimer.h"

#include <boost/foreach.hpp>

//#define DEBUGINFO
#define DEBUGINFO if(0)

//#define TASKINFO
#define TASKINFO if(0)

using namespace boost;

namespace Signal {
namespace Processing {


Step::Step(OperationDesc::Ptr operation_desc)
    :
        not_started_(Intervals::Intervals_ALL),
        operation_desc_(operation_desc)
{
}


Signal::OperationDesc::Ptr Step::
        get_crashed() const
{
    return died_;
}


void Step::
        mark_as_crashed()
{
    if (died_)
        return;

    DEBUGINFO TaskInfo ti(boost::format("Marking step \"%s\" as crashed") % operation_name());

    died_ = operation_desc_;
    operation_desc_ = Signal::OperationDesc::Ptr(new Test::TransparentOperationDesc);
    operations_.clear ();

    Signal::OperationDesc::Ptr died = died_;
    bool was_locked = !readWriteLock ()->tryLockForWrite ();
    readWriteLock ()->unlock ();

    // Don't use 'this' while unlocked.
    died->deprecateCache(Signal::Interval::Interval_ALL);

    if (was_locked && !readWriteLock ()->tryLockForWrite (VolatilePtr_lock_timeout_ms))
        BOOST_THROW_EXCEPTION(LockFailed()
                              << typename LockFailed::timeout_value(VolatilePtr_lock_timeout_ms)
                              << Backtrace::make());
}


std::string Step::
        operation_name ()
{
    return (operation_desc_?read1(operation_desc_)->toString ().toStdString ():"(no operation)");
}


Intervals Step::
        currently_processing() const
{
    Intervals I;

    BOOST_FOREACH(RunningTaskMap::value_type ti, running_tasks)
    {
        I |= ti.second;
    }

    return I;
}


Intervals Step::
        deprecateCache(Intervals deprecated)
{
    if (deprecated == Interval::Interval_ALL) {
        cache_.reset ();
        operations_.clear ();
    }

    if (operation_desc_ && deprecated) {
        OperationDesc::ReadPtr o(operation_desc_);

        Intervals A;
        BOOST_FOREACH(const Interval& i, deprecated) {
            A |= o->affectedInterval(i);
        }
        deprecated = A;
    }

    DEBUGINFO TaskInfo(format("Step %1%. Deprecate %2%")
              % (operation_desc_?read1(operation_desc_)->toString ().toStdString ():"(no operation)")
              % deprecated);

    not_started_ |= deprecated;

    return deprecated;
}


Intervals Step::
        not_started() const
{
    return not_started_;
}


Intervals Step::
        out_of_date() const
{
    return not_started_ | currently_processing();
}


Operation::Ptr Step::
        operation(ComputingEngine::Ptr ce)
{
    gc();

    ComputingEngine::WeakPtr wp(ce);
    OperationMap::iterator oi = operations_.find (wp);

    if (oi != operations_.end ())
    {
        return oi->second;
    }

    Operation::Ptr o = read1(operation_desc_)->createOperation (ce.get ());
    operations_[wp] = o;

    return o;
}


OperationDesc::Ptr Step::
        operation_desc () const
{
    return operation_desc_;
}


void Step::
        registerTask(Task* taskid, Interval expected_output)
{
    TASKINFO TaskInfo ti(format("Step %1%. Starting %2%")
              % operation_name()
              % expected_output);
    running_tasks[taskid] = expected_output;
    not_started_ -= expected_output;
}


void Step::
        finishTask(Task* taskid, pBuffer result)
{
    Interval result_interval;
    if (result)
        result_interval = result->getInterval ();

    TASKINFO TaskInfo ti(format("Step %1%. Finish %2%")
              % operation_name()
              % result_interval);

    if (result) {
        if (!cache_)
            cache_.reset(new SinkSource(result->number_of_channels ()));

        // Result must have the same number of channels and sample rate as previous cache.
        // Call deprecateCache(Interval::Interval_ALL) to erase the cache when chainging number of channels or sample rate.
        cache_->put (result);
    }

    int C = running_tasks.count (taskid);
    if (C!=1) {
        TaskInfo("C = %d, taskid = %x", C, taskid);
        EXCEPTION_ASSERTX( running_tasks.count (taskid)==1, "Could not find given task");
    }

    Intervals expected_output = running_tasks[ taskid ];

    Intervals update_miss = expected_output - result_interval;
    not_started_ |= update_miss;

    if (!expected_output) {
        TaskInfo(format("The task was not recognized. %1% on %2%")
                 % result_interval
                 % operation_name());
    } else if (!result_interval) {
        TASKINFO TaskInfo(format("The task was cancelled. Restoring %1% for %2%")
                 % update_miss
                 % operation_name());
    } else {
        if (update_miss) {
            TaskInfo(format("These samples were supposed to be updated by the task but missed: %1% by %2%")
                     % update_miss
                     % operation_name());
        }
        if (result_interval - expected_output) {
            // These samples were not supposed to be updated by the task but were calculated anyway
            TaskInfo(format("Unexpected extras: %1% = (%2%) - (%3%) from %4%")
                     % (result_interval - expected_output)
                     % result_interval
                     % expected_output
                     % operation_name());

            // The samples are still marked as invalid. Would need to remove the
            // extra calculated samples from not_started_ but that would fail
            // in a situation where deprecatedCache is called after the task has
            // been created. So not_started_ can't be modified here (unless calls
            // to deprecatedCache were tracked).
        }
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


pBuffer Step::
        readFixedLengthFromCache(Interval I)
{
    return cache_ ? cache_->readFixedLength (I) : pBuffer();
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

#include "signal/operation-basic.h"

namespace Signal {
namespace Processing {

void Step::
        test()
{
    // It should keep a cache for a signal processing step (defined by an OpertionDesc).
    //
    // The cache description should contain information about what's out_of_date
    // and what's currently being updated.
    {
        // Create an OperationDesc
        pBuffer b(new Buffer(Interval(60,70), 40, 7));
        for (unsigned c=0; c<b->number_of_channels (); ++c)
        {
            float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
            for (int i=0; i<b->number_of_samples (); ++i)
                p[i] = c + 1+i/(float)b->number_of_samples ();
        }

        // Create a Step
        Step s((OperationDesc::Ptr()));

        // It should contain information about what's out_of_date and what's currently being updated.
        s.registerTask(0, b->getInterval ());
        EXCEPTION_ASSERT_EQUALS(s.not_started (), ~Intervals(b->getInterval ()));
        EXCEPTION_ASSERT_EQUALS(s.out_of_date(), Intervals::Intervals_ALL);
        s.finishTask(0, b);
        EXCEPTION_ASSERT_EQUALS(s.out_of_date(), ~Intervals(b->getInterval ()));

        EXCEPTION_ASSERT( *b == *s.readFixedLengthFromCache (b->getInterval ()) );
    }

    // A crashed signal processing step should behave as a transparent operation.
    {
        OperationDesc::Ptr silence(new Signal::OperationSetSilent(Signal::Interval(2,3)));
        Step s(silence);
        EXCEPTION_ASSERT(!s.get_crashed ());
        EXCEPTION_ASSERT(!dynamic_cast<volatile Test::TransparentOperationDesc*>(s.operation_desc ().get ()));
        EXCEPTION_ASSERT(!dynamic_cast<volatile Test::TransparentOperation*>(s.operation (Signal::ComputingEngine::Ptr()).get ()));
        s.mark_as_crashed ();
        EXCEPTION_ASSERT(s.get_crashed ());
        EXCEPTION_ASSERT(dynamic_cast<volatile Test::TransparentOperationDesc*>(s.operation_desc ().get ()));
        EXCEPTION_ASSERT(dynamic_cast<volatile Test::TransparentOperation*>(s.operation (Signal::ComputingEngine::Ptr()).get ()));
    }
}


} // namespace Processing
} // namespace Signal
