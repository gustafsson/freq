#include "step.h"
#include "test/operationmockups.h"

#include "tasktimer.h"
#include "log.h"

#include <boost/foreach.hpp>

//#define DEBUGINFO
#define DEBUGINFO if(0)

//#define TASKINFO
#define TASKINFO if(0)

using namespace boost;

namespace Signal {
namespace Processing {


Step::Step(OperationDesc::ptr operation_desc)
    :
        not_started_(Intervals::Intervals_ALL),
        operation_desc_(operation_desc)
{
}


Signal::OperationDesc::ptr Step::
        get_crashed() const
{
    return died_;
}


Signal::Processing::IInvalidator::ptr Step::
        mark_as_crashed_and_get_invalidator()
{
    if (died_)
        return Signal::Processing::IInvalidator::ptr();

    DEBUGINFO TaskInfo ti(boost::format("Marking step \"%s\" as crashed") % operation_name());

    died_ = operation_desc_;
    operation_desc_ = Signal::OperationDesc::ptr(new Test::TransparentOperationDesc);

    return died_.read ()->getInvalidator ();
}


std::string Step::
        operation_name ()
{
    return (operation_desc_?operation_desc_.read ()->toString ().toStdString ():"(no operation)");
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
        cache_.clear ();
    }

    if (operation_desc_ && deprecated) {
        auto o = operation_desc_.read ();

        Intervals A;
        BOOST_FOREACH(const Interval& i, deprecated) {
            A |= o->affectedInterval(i);
        }
        deprecated = A;
    }

    DEBUGINFO TaskInfo(format("Step %1%. Deprecate %2%")
              % (operation_desc_?operation_desc_.read ()->toString ().toStdString ():"(no operation)")
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


OperationDesc::ptr Step::
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
        // Result must have the same number of channels and sample rate as previous cache.
        // Call deprecateCache(Interval::Interval_ALL) to erase the cache when chainging number of channels or sample rate.
        cache_.put (result);
    }

    int matched_task = running_tasks.count (taskid);
    if (1 != matched_task) {
        Log("C = %d, taskid = %x on %s") % matched_task % taskid % operation_name ();
        EXCEPTION_ASSERT_EQUALS( 1, matched_task );
    }

    Intervals expected_output = running_tasks[ taskid ];

    Intervals update_miss = expected_output - result_interval;
    not_started_ |= update_miss;

    if (!result) {
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
    wait_for_tasks_.notify_all ();
}


void Step::
        sleepWhileTasks(Step::ptr::read_ptr& step, int sleep_ms)
{
    DEBUGINFO TaskTimer tt(boost::format("sleepWhileTasks %d") % step->running_tasks.size ());

    // The caller keeps a lock that is released while waiting
    // Wait in a while-loop to cope with spurious wakeups
    if (sleep_ms < 0)
    {
        while (!step->running_tasks.empty ())
            step->wait_for_tasks_.wait(step);
    }
    else
    {
        std::chrono::milliseconds ms(sleep_ms);

        while (!step->running_tasks.empty ())
            if (std::cv_status::timeout == step->wait_for_tasks_.wait_for (step, ms))
                return;
    }
}


pBuffer Step::
        readFixedLengthFromCache(Interval I) const
{
    return cache_.read (I);
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
        Step s((OperationDesc::ptr()));

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
        OperationDesc::ptr silence(new Signal::OperationSetSilent(Signal::Interval(2,3)));
        Step s(silence);
        EXCEPTION_ASSERT(!s.get_crashed ());
        EXCEPTION_ASSERT(s.operation_desc ());
        EXCEPTION_ASSERT(s.operation_desc ().read ()->createOperation (0));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperationDesc*>(s.operation_desc ().raw ()));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperation*>(s.operation_desc ().read ()->createOperation (0).raw ()));
        s.mark_as_crashed_and_get_invalidator ();
        EXCEPTION_ASSERT(s.get_crashed ());
        EXCEPTION_ASSERT(s.operation_desc ());
        EXCEPTION_ASSERT(s.operation_desc ().read ()->createOperation (0));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperationDesc*>(s.operation_desc ().raw ()));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperation*>(s.operation_desc ().read ()->createOperation (0).raw ()));
    }
}


} // namespace Processing
} // namespace Signal
