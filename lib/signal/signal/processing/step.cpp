#include "step.h"
#include "test/operationmockups.h"

#include "tasktimer.h"
#include "log.h"

#include <boost/foreach.hpp>

//#define DEBUGINFO
#define DEBUGINFO if(0)

//#define TASKINFO
#define TASKINFO if(0)

//#define FINISHTASKINFO
#define FINISHTASKINFO if(0)

//#define LOG_PURGED_CACHES
#define LOG_PURGED_CACHES if(0)

using namespace boost;

namespace Signal {
namespace Processing {


Step::Step(OperationDesc::ptr operation_desc)
    :
        cache_(new Cache),
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

    DEBUGINFO Log("Step: marking \"%s\" as crashed") % operation_name();

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
    running_tasks.clear ();
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
        // to be used with not_started() to determine what to start new tasks on
        I |= ti.valid_output;
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

    DEBUGINFO Log("Step: deprecateCache %2% | %3% on %1%")
              % operation_name()
              % deprecated
              % not_started();

    cache_->invalidate_samples(deprecated);

    Signal::Intervals not_deprecated = ~deprecated;
    for (RunningTaskMap::value_type& v : running_tasks)
        v.valid_output &= not_deprecated;

    return deprecated;
}


size_t Step::
        purge(Signal::Intervals still_needed)
{
    auto cache = cache_.write ();
    int C = cache->num_channels ();
    Signal::Intervals P = cache->purge (still_needed);
    if (P)
        LOG_PURGED_CACHES Log("Step: discarding %s, only need %s for %s") % P % still_needed % operation_name();
    return P.count () * C;
}


Intervals Step::
        not_started() const
{
    return ~cache_.read ()->samplesDesc() & ~currently_processing();
}


OperationDesc::ptr Step::
        operation_desc (const_ptr step)
{
    return step.raw ()->operation_desc_;
}


int Step::
        registerTask(Step::ptr::write_ptr&& step, Signal::Interval expected_output)
{
    return registerTask (step, expected_output);
}


int Step::
        registerTask(Step::ptr::write_ptr& step, Interval expected_output)
{
    TASKINFO Log("Step registerTask %2% on %1%")
              % step->operation_name()
              % expected_output;

    ++step->task_counter_;
    if (0 == step->task_counter_)
        ++step->task_counter_;
    int taskid = step->task_counter_;

    step->running_tasks.emplace_back(taskid,expected_output);

    // enforce ordering of tasks, wait for previous tasks to finish if they affect the same output
    auto ready = [&](){
        for (RunningTaskMap::value_type& v : step->running_tasks)
        {
            if (v.id == taskid)
                break;
            if (v.expected_output & expected_output)
                return false;
        }
        return true;
    };

    step->wait_for_tasks_.wait(step, ready);

    return taskid;
}


void Step::
        finishTask(Step::ptr step, int taskid, pBuffer result)
{
    FINISHTASKINFO Log("Step finishTask %2% on %1%")
              % step.raw ()->operation_name()
              % result->getInterval ();

    auto self = step.write ();

    Intervals valid_output;
    for(auto i = self->running_tasks.begin(); i != self->running_tasks.end(); i++)
        if (i->id == taskid)
        {
            valid_output = i->valid_output;
            self->running_tasks.erase ( i );
            break;
        }

    self.unlock ();

    if (!valid_output)
        result.reset ();

    if (result)
      {
        // Result must have the same number of channels and sample rate as previous cache.
        // Call deprecateCache(Interval::Interval_ALL) to erase the cache when chainging number of channels or sample rate.

        TASKINFO if (valid_output - step.raw ()->cache_->allocated())
            Log("Step allocating for %s") % step.raw ()->operation_name ();

        if (valid_output == result->getInterval ())
            step.raw ()->cache_->put (result);
        else
            for (auto i : valid_output)
              {
                pBuffer b(new Buffer(i, result->sample_rate (), result->number_of_channels ()));
                *b |= *result;
                step.raw ()->cache_->put (result);
              }
      }

    step.raw ()->wait_for_tasks_.notify_all ();
}


bool Step::
        sleepWhileTasks(Step::ptr::read_ptr& step, int sleep_ms)
{
    DEBUGINFO TaskTimer tt(boost::format("Step: sleepWhileTasks %d") % step->running_tasks.size ());

    // The caller keeps a lock that is released while waiting
    // Wait in a while-loop to cope with spurious wakeups
    if (sleep_ms < 0)
    {
        step->wait_for_tasks_.wait(step, [&](){return step->running_tasks.empty ();});
    }
    else
    {
        std::chrono::milliseconds ms(sleep_ms);

        step->wait_for_tasks_.wait_for (step, ms, [&](){return step->running_tasks.empty ();});
    }

    return step->running_tasks.empty ();
}


bool Step::
        sleepWhileTasks(Step::ptr::read_ptr&& step, int sleep_ms)
{
    return sleepWhileTasks(step, sleep_ms);
}


shared_state<const Signal::Cache> Step::
        cache(const_ptr step)
{
    return step.raw ()->cache_;
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
        int taskid = Step::registerTask(s.write (),b->getInterval ());
        EXCEPTION_ASSERT_EQUALS(s->not_started (), ~Intervals(b->getInterval ()));
        EXCEPTION_ASSERT_EQUALS(Step::cache (s)->samplesDesc(), Intervals());
        Step::finishTask(s, taskid, b);
        EXCEPTION_ASSERT_EQUALS(Step::cache (s)->samplesDesc(), Intervals(b->getInterval ()));

        EXCEPTION_ASSERT( *b == *Step::cache (s)->read(b->getInterval ()) );
    }

    // A crashed signal processing step should behave as a transparent operation.
    {
        OperationDesc::ptr silence(new Signal::OperationSetSilent(Signal::Interval(2,3)));
        Step::ptr s(new Step(silence));
        EXCEPTION_ASSERT(!s->get_crashed ());
        EXCEPTION_ASSERT(Step::operation_desc (s));
        EXCEPTION_ASSERT(Step::operation_desc (s).read ()->createOperation (0));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperationDesc*>(Step::operation_desc (s).raw ()));
        EXCEPTION_ASSERT(!dynamic_cast<Test::TransparentOperation*>(Step::operation_desc (s).read ()->createOperation (0).get ()));
        s->mark_as_crashed_and_get_invalidator ();
        EXCEPTION_ASSERT(s->get_crashed ());
        EXCEPTION_ASSERT(Step::operation_desc (s));
        EXCEPTION_ASSERT(Step::operation_desc (s).read ()->createOperation (0));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperationDesc*>(Step::operation_desc (s).raw ()));
        EXCEPTION_ASSERT(dynamic_cast<Test::TransparentOperation*>(Step::operation_desc (s).read ()->createOperation (0).get ()));
    }
}


} // namespace Processing
} // namespace Signal
