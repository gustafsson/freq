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
        cache_(new Cache),
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


void Step::
        undie()
{
    if (died_)
        operation_desc_ = died_;
    died_.reset ();
    cache_->clear ();
    not_started_ = Signal::Intervals::Intervals_ALL;
}


std::string Step::
        operation_name () const
{
    Signal::OperationDesc::ptr operation_desc = this->operation_desc_;
    return (operation_desc?operation_desc.raw ()->toString ().toStdString ():"(no operation)");
}


Intervals Step::
        currently_processing() const
{
    Intervals I;

    for (RunningTaskMap::value_type ti : running_tasks)
    {
        I |= ti.second;
    }

    return I;
}


Intervals Step::
        deprecateCache(Intervals deprecated)
{
    // Could remove all allocated cache memory here if the entire interval is deprecated.
    // But it is highly likely that it will be required again very soon, so don't bother.

    if (operation_desc_ && deprecated) {
        auto o = operation_desc_.read ();

        Intervals A;
        for (const Interval& i : deprecated) {
            A |= o->affectedInterval(i);
        }
        deprecated = A;
    }

    DEBUGINFO TaskInfo(format("Step::deprecateCache %2% | %3% on %1%")
              % operation_name()
              % deprecated
              % not_started_);

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
    Intervals c = currently_processing();
    //DEBUGINFO TaskInfo(boost::format("Step::out_of_date: %s = %s | %s in %s")
    //         % (not_started_ | c) % not_started_ % c % operation_name());
    return not_started_ | c;
}


OperationDesc::ptr Step::
        operation_desc () const
{
    return operation_desc_;
}


int Step::
        registerTask(Interval expected_output)
{
    TASKINFO TaskInfo ti(format("Step::registerTask %2% on %1%")
              % operation_name()
              % expected_output);

    ++task_counter_;
    if (0 == task_counter_)
        ++task_counter_;
    int taskid = task_counter_;

    running_tasks[taskid] = expected_output;
    not_started_ -= expected_output;
    return taskid;
}


void Step::
        finishTask(Step::ptr step, int taskid, pBuffer result)
{
    Interval result_interval;
    if (result)
        result_interval = result->getInterval ();

    TASKINFO TaskInfo ti(format("Step::finishTask %2% on %1%")
              % step.raw ()->operation_name()
              % result_interval);

    if (result) {
        // Result must have the same number of channels and sample rate as previous cache.
        // Call deprecateCache(Interval::Interval_ALL) to erase the cache when chainging number of channels or sample rate.
        step.raw ()->cache_->put (result);
    }

    auto self = step.write ();
    int matched_task = self->running_tasks.count (taskid);
    if (1 != matched_task) {
        Log("C = %d, taskid = %x on %s") % matched_task % taskid % step.raw ()->operation_name ();
        EXCEPTION_ASSERT_EQUALS( 1, matched_task );
    }

    Intervals expected_output = self->running_tasks[ taskid ];

    Intervals update_miss = expected_output - result_interval;
    self->not_started_ |= update_miss;

    if (!result) {
        TASKINFO TaskInfo(format("The task was cancelled. Restoring %1% for %2%")
                 % update_miss
                 % step.raw ()->operation_name());
    } else {
        if (update_miss) {
            TaskInfo(format("These samples were supposed to be updated by the task but missed: %1% by %2%")
                     % update_miss
                     % step.raw ()->operation_name());
        }
        if (result_interval - expected_output) {
            // These samples were not supposed to be updated by the task but were calculated anyway
            TaskInfo(format("Unexpected extras: %1% = (%2%) - (%3%) from %4%")
                     % (result_interval - expected_output)
                     % result_interval
                     % expected_output
                     % step.raw ()->operation_name());

            // The samples are still marked as invalid. Would need to remove the
            // extra calculated samples from not_started_ but that would fail
            // in a situation where deprecatedCache is called after the task has
            // been created. So not_started_ can't be modified here (unless calls
            // to deprecatedCache were tracked).
        }
    }

    self->running_tasks.erase ( taskid );

    self.unlock ();

    step.raw ()->wait_for_tasks_.notify_all ();
}


bool Step::
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
                return false;
    }

    return step->running_tasks.empty ();
}


bool Step::
        sleepWhileTasks(Step::ptr::read_ptr&& step, int sleep_ms)
{
    return sleepWhileTasks(step, sleep_ms);
}


pBuffer Step::
        readFixedLengthFromCache(Step::const_ptr ptr, Interval I)
{
    return ptr.raw ()->cache_.read ()->read (I);
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
        Step::ptr s2( new Step(OperationDesc::ptr()));

        shared_state<Step>::weak_ptr ws = s2;
        Step::ptr s = ws.lock ();

        // It should contain information about what's out_of_date and what's currently being updated.
        int taskid = s->registerTask(b->getInterval ());
        EXCEPTION_ASSERT_EQUALS(s->not_started (), ~Intervals(b->getInterval ()));
        EXCEPTION_ASSERT_EQUALS(s->out_of_date(), Intervals::Intervals_ALL);
        Step::finishTask(s, taskid, b);
        EXCEPTION_ASSERT_EQUALS(s->out_of_date(), ~Intervals(b->getInterval ()));

        EXCEPTION_ASSERT( *b == *Step::readFixedLengthFromCache (s, b->getInterval ()) );
    }

    // A crashed signal processing step should behave as a transparent operation.
    {
        OperationDesc::ptr silence(new Signal::OperationSetSilent(Signal::Interval(2,3)));
        Step s(silence);
        EXCEPTION_ASSERT(!s.get_crashed ());
        EXCEPTION_ASSERT(s.operation_desc ());
        EXCEPTION_ASSERT(s.operation_desc ().read ()->createOperation (0));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperationDesc*>(s.operation_desc ().raw ()));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperation*>(s.operation_desc ().read ()->createOperation (0).get ()));
        s.mark_as_crashed_and_get_invalidator ();
        EXCEPTION_ASSERT(s.get_crashed ());
        EXCEPTION_ASSERT(s.operation_desc ());
        EXCEPTION_ASSERT(s.operation_desc ().read ()->createOperation (0));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperationDesc*>(s.operation_desc ().raw ()));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperation*>(s.operation_desc ().read ()->createOperation (0).get ()));
    }
}


} // namespace Processing
} // namespace Signal
