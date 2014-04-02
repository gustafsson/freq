#include "targetneeds.h"
#include "step.h"
#include "bedroom.h"

#include "timer.h"
#include "tasktimer.h"

#include <boost/date_time/posix_time/posix_time.hpp>

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetNeeds::
        TargetNeeds(Step::ptr::weak_ptr step, INotifier::weak_ptr notifier)
    :
      step_(step),
      work_center_(Signal::Interval::IntervalType_MIN),
      preferred_update_size_(Signal::Interval::IntervalType_MAX),
      notifier_(notifier)
{
}


TargetNeeds::
        ~TargetNeeds()
{
    updateNeeds(Signal::Intervals());
}


void TargetNeeds::
        updateNeeds
        (
        Signal::Intervals needed_samples,
        Signal::IntervalType center,
        Signal::IntervalType preferred_update_size,
        int prio
        )
{
    EXCEPTION_ASSERT_LESS( 0, preferred_update_size_ );
    Signal::Intervals not_started;
    if (Step::ptr step = step_.lock ()) {
        not_started = step.read ()->not_started ();
    }

    // got news if something new and deprecated is needed
    bool got_news = (needed_samples - needed_samples_) & not_started;

    DEBUG_INFO {
        TaskInfo ti(boost::format("news = %s, needed_samples = %s, not_started = %s")
                 % got_news % needed_samples % not_started);
        if (needed_samples != needed_samples_)
            TaskInfo(boost::format("needed_samples_ = %s") % needed_samples_);
    }

    needed_samples_ = needed_samples;

    ptime now = microsec_clock::local_time();
    last_request_ = now + time_duration(0,0,prio);

    work_center_ = center;

    preferred_update_size_ = preferred_update_size;

    if (got_news) {
        INotifier::ptr notifier = notifier_.lock ();
        if (notifier)
            notifier->wakeup();
    }
}


void TargetNeeds::
        deprecateCache(Signal::Intervals invalidate)
{
    if (!invalidate)
        return;

    DEBUG_INFO TaskInfo(boost::format("invalidate = %s") % invalidate);

    if (Step::ptr step = step_.lock ())
        step.write ()->deprecateCache(invalidate);

    if (invalidate & needed_samples_) {
        INotifier::ptr notifier = notifier_.lock ();
        if (notifier)
            notifier->wakeup();
    }
}


Step::ptr::weak_ptr TargetNeeds::
        step() const
{
    return step_;
}


boost::posix_time::ptime TargetNeeds::
        last_request() const
{
    return last_request_;
}


Signal::IntervalType TargetNeeds::
        work_center() const
{
    return work_center_;
}


Signal::IntervalType TargetNeeds::
        preferred_update_size() const
{
    return preferred_update_size_;
}


Signal::Intervals TargetNeeds::
        not_started() const
{
    Signal::Intervals not_started;
    Step::ptr step = step_.lock ();
    if (step)
        not_started = step.read ()->not_started();

    return needed_samples_ & not_started;
}


Signal::Intervals TargetNeeds::
        out_of_date() const
{
    Signal::Intervals out_of_date;
    Step::ptr step = step_.lock ();
    if (step)
        out_of_date = step.read ()->out_of_date();

    return needed_samples_ & out_of_date;
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
        sleep(TargetNeeds::const_ptr self, int sleep_ms)
{
    Timer t;

    Step::ptr pstep = self.raw ()->step_.lock();

    if (!pstep)
        return false;

    for (;;)
    {
        auto step = pstep.read ();

        if (!(self->needed_samples_ & step->out_of_date ()))
            return true;

        Step::sleepWhileTasks (step, left(t, sleep_ms));

        if (!(self->needed_samples_ & step->out_of_date ()))
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
        step.write ()->registerTask(0, initial_valid.spannedInterval ());

        EXCEPTION_ASSERT_EQUALS( step.read ()->out_of_date(), Signal::Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( step.read ()->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( target_needs.read ()->out_of_date(), Signal::Interval() );
        target_needs.write ()->updateNeeds(Signal::Interval(-15,5));
        EXCEPTION_ASSERT_EQUALS( step.read ()->out_of_date(), Signal::Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( step.read ()->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( target_needs.read ()->out_of_date(), Signal::Interval(-15,5) );
        EXCEPTION_ASSERT_EQUALS( target_needs.read ()->not_started(), Signal::Interval(-15,0) );
    }

    // It should not wakeup the bedroom if nothing has changed
    {
        // Note; this is more helpful to do a less noisy debugging than to increase any performance
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        // Validate a bit Signal::Interval(0,10) of the step
        step.write ()->registerTask(0, Signal::Interval(0,10));
        step.write ()->finishTask(0, Signal::pBuffer(new Signal::Buffer(Signal::Interval(0,10),1,1)));

        TargetNeeds::ptr target_needs( new TargetNeeds(step, notifier) );

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something deprecated");
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->updateNeeds(Signal::Interval(-15,5));
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep for too long");
            target_needs.write ()->updateNeeds(Signal::Interval(-15,5));

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
            target_needs.write ()->updateNeeds(Signal::Interval(-15,4));
            bed.sleep (2);
            float T = t.elapsed ();
            EXCEPTION_ASSERT_LESS(2e-3, T); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep for too long 3");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->updateNeeds(Signal::Interval(-15,6));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep for too long 4");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->deprecateCache(Signal::Intervals(6,7));
            target_needs.write ()->updateNeeds(Signal::Interval(-15,6),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX);
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new");
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->updateNeeds(Signal::Interval(-15,7));
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new 2");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->deprecateCache(Signal::Intervals(6,7));
            target_needs.write ()->updateNeeds(Signal::Interval(-15,7),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX);
            bed.sleep (2);
        }

        {
            TRACE_PERF("Should not sleep since updateNeeds needs something new 3");

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            target_needs.write ()->deprecateCache(Signal::Intervals(5,7));
            target_needs.write ()->updateNeeds(Signal::Interval(-15,7),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX);
            bed.sleep (2);
        }
    }
}

} // namespace Processing
} // namespace Signal
