#include "targetneeds.h"
#include "bedroom.h"

#include "timer.h"
#include "tasktimer.h"
#include "log.h"

#include <boost/date_time/posix_time/posix_time.hpp>

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetNeeds::
        TargetNeeds(shared_state<Step>::weak_ptr step, INotifier::weak_ptr notifier)
    :
      state_(new State),
      step_(step),
      notifier_(notifier)
{
    auto state = state_.write ();
    state->work_center = Interval::IntervalType_MIN;
    state->preferred_update_size = Interval::IntervalType_MAX;
}


TargetNeeds::
        ~TargetNeeds()
{
}


void TargetNeeds::
        updateNeeds
        (
        const Signal::Intervals& needed_samples,
        Signal::IntervalType center,
        Signal::IntervalType preferred_update_size,
        int prio
        )
{
    EXCEPTION_ASSERT_LESS( 0, preferred_update_size );

    auto state = state_.write ();
    Intervals new_samples = needed_samples - state->needed_samples;
    ptime now = microsec_clock::local_time();
    state->last_request = now + time_duration(0,0,prio);
    state->needed_samples = needed_samples;
    state->work_center = center;
    state->preferred_update_size = preferred_update_size;

    state.unlock ();

    DEBUG_INFO TaskInfo(boost::format("needed_samples = %s") % needed_samples);

    Step::const_ptr step = step_.lock ();
    INotifier::ptr notifier = notifier_.lock ();
    if (new_samples && step && notifier)
    {
        // got news if something new and deprecated is needed
        if (new_samples & step.read ()->not_started ())
            notifier->wakeup();
    }
}


void TargetNeeds::
        deprecateCache(const Intervals& invalidate) const
{
    if (!invalidate)
        return;

    DEBUG_INFO TaskInfo(boost::format("invalidate = %s") % invalidate);

    if (Step::ptr step = step_.lock ())
        step->deprecateCache(invalidate);

    if (INotifier::ptr notifier = notifier_.lock ())
        if (invalidate & state_.read ()->needed_samples)
            notifier->wakeup();
}


Step::ptr::weak_ptr TargetNeeds::
        step() const
{
    return step_;
}


boost::posix_time::ptime TargetNeeds::
        last_request() const
{
    return state_->last_request;
}


Signal::IntervalType TargetNeeds::
        work_center() const
{
    return state_->work_center;
}


Signal::IntervalType TargetNeeds::
        preferred_update_size() const
{
    return state_->preferred_update_size;
}


Signal::Intervals TargetNeeds::
        not_started() const
{
    Signal::Intervals not_started;
    Step::ptr step = step_.lock ();
    if (step)
        not_started = step.read ()->not_started();

    return needed() & not_started;
}


Signal::Intervals TargetNeeds::
        needed() const
{
    return state_->needed_samples;
}


TargetNeeds::State TargetNeeds::
        state() const
{
    return *state_.get ();
}


Signal::Intervals TargetNeeds::
        out_of_date() const
{
    Signal::Intervals out_of_date;
    Step::ptr step = step_.lock ();
    if (step)
        out_of_date = step.read ()->out_of_date();

    Signal::Intervals needed = this->needed ();
    DEBUG_INFO Log("TargetNeeds::out_of_date: %s = %s & %s")
            % (needed & out_of_date) % needed % out_of_date;
    return needed & out_of_date;
}


int left(const Timer& t, int sleep_ms) {
    if (sleep_ms < 0)
        return sleep_ms;

    float elapsed = t.elapsed ();
    if (elapsed > sleep_ms/1000.f)
        elapsed = sleep_ms/1000.f;
    int left = elapsed*1000 - sleep_ms;
    return left;
}


bool TargetNeeds::
        sleep(int sleep_ms) const
{
    Timer t;

    Step::ptr pstep = step_.lock();

    if (!pstep)
        return false;

    for (;;)
    {
        auto step = pstep.read ();

        if (!(needed() & step->out_of_date ()))
            return true;

        Step::sleepWhileTasks (step, left(t, sleep_ms));

        if (!(needed() & step->out_of_date ()))
            return true;

        step.unlock ();

        if (0 <= sleep_ms && 0 == left(t, sleep_ms))
            return false;

        std::this_thread::sleep_for (std::chrono::milliseconds(1));
    }
}

} // namespace Processing
} // namespace Signal

#include "dag.h"
#include "targets.h"
#include "bedroomnotifier.h"
#include "trace_perf.h"

using namespace std;

namespace Signal {
namespace Processing {

void TargetNeeds::
        test()
{
    // It should describe what needs to be computed for a target.
    {
        Bedroom::ptr bedroom(new Bedroom);
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));

        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        TargetNeeds::ptr target_needs( new TargetNeeds(step, notifier) );

        Signal::Intervals initial_valid(0,60);
        int taskid = step.write ()->registerTask(initial_valid.spannedInterval ());
        (void)taskid; // discard

        EXCEPTION_ASSERT_EQUALS( step.read ()->out_of_date(), Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( step.read ()->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( target_needs->out_of_date(), Interval() );
        target_needs->updateNeeds(Interval(-15,5));
        EXCEPTION_ASSERT_EQUALS( step.read ()->out_of_date(), Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( step.read ()->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( target_needs->out_of_date(), Interval(-15,5) );
        EXCEPTION_ASSERT_EQUALS( target_needs->not_started(), Interval(-15,0) );
    }

    // It should not wakeup the bedroom if nothing has changed
    {
        // Note; this is more helpful to do a less noisy debugging than to increase any performance
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        // Validate a bit Signal::Interval(0,10) of the step
        int taskid = step.write ()->registerTask(Signal::Interval(0,10));
        Step::finishTask(step, taskid, pBuffer(new Buffer(Interval(0,10),1,1)));

        TargetNeeds::ptr target_needs( new TargetNeeds(step, notifier) );

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something deprecated");
            Bedroom::Bed bed = bedroom->getBed();
            target_needs->updateNeeds(Interval(-15,5));
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep for too long");
            target_needs->updateNeeds(Signal::Interval(-15,5));

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            bed.sleep (2);
            float T = t.elapsed ();
            EXCEPTION_ASSERT_LESS(2e-3, T); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep for too long 2");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs->updateNeeds(Signal::Interval(-15,4));
            bed.sleep (2);
            float T = t.elapsed ();
            EXCEPTION_ASSERT_LESS(2e-3, T); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep for too long 3");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs->updateNeeds(Signal::Interval(-15,6));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep for too long 4");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs->deprecateCache(Signal::Intervals(6,7));
            target_needs->updateNeeds(Signal::Interval(-15,6),
                                      Signal::Interval::IntervalType_MIN,
                                      Signal::Interval::IntervalType_MAX);
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new");

            Bedroom::Bed bed = bedroom->getBed();
            target_needs->updateNeeds(Signal::Interval(-15,7));
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new 2");

            Bedroom::Bed bed = bedroom->getBed();
            target_needs->deprecateCache(Signal::Intervals(6,7));
            target_needs->updateNeeds(Signal::Interval(-15,7),
                                      Signal::Interval::IntervalType_MIN,
                                      Signal::Interval::IntervalType_MAX);
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new 3");

            Bedroom::Bed bed = bedroom->getBed();
            target_needs->deprecateCache(Intervals(5,7));
            target_needs->updateNeeds(Interval(-15,7),
                                      Interval::IntervalType_MIN,
                                      Interval::IntervalType_MAX);
            bed.sleep (2);
        }
    }
}

} // namespace Processing
} // namespace Signal
