#include "targetneeds.h"
#include "step.h"
#include "bedroom.h"

#include "timer.h"

#include <boost/date_time/posix_time/posix_time.hpp>

//#define DEBUG_INFO
#define DEBUG_INFO if(0)

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetNeeds::
        TargetNeeds(Step::WeakPtr step, Bedroom::WeakPtr bedroom)
    :
      step_(step),
      work_center_(Signal::Interval::IntervalType_MIN),
      preferred_update_size_(Signal::Interval::IntervalType_MAX),
      bedroom_(bedroom)
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
        Signal::Intervals invalidate,
        int prio
        )
{
    EXCEPTION_ASSERT_LESS( 0, preferred_update_size_ );
    Signal::Intervals not_started;
    if (Step::Ptr pstep = step_.lock ()) {
        Step::WritePtr step(pstep);
        not_started = step->not_started ();

        if (invalidate)
            step->deprecateCache(invalidate);
    }

    // got news if something new is needed
    bool got_news = needed_samples - needed_samples_;
    // or if something new is invalidated that is also needed
    got_news |= (invalidate - not_started) & needed_samples;

    DEBUG_INFO {
        TaskInfo ti(boost::format("news = %s, needed_samples = %s, not_started = %s")
                 % got_news % needed_samples % not_started);
        if (needed_samples != needed_samples_)
            TaskInfo(boost::format("needed_samples_ = %s") % needed_samples_);
        if (invalidate)
            TaskInfo(boost::format("invalidate = %s") % invalidate);
    }

    needed_samples_ = needed_samples;

    ptime now = microsec_clock::local_time();
    last_request_ = now + time_duration(0,0,prio);

    work_center_ = center;

    preferred_update_size_ = preferred_update_size;

    if (got_news) {
        Bedroom::Ptr bedroom = bedroom_.lock ();
        if (bedroom)
            bedroom->wakeup();
    }
}


Step::WeakPtr TargetNeeds::
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
    Step::Ptr step = step_.lock ();
    if (step)
        not_started = read1(step)->not_started();

    return needed_samples_ & not_started;
}


Signal::Intervals TargetNeeds::
        out_of_date() const
{
    Signal::Intervals out_of_date;
    Step::Ptr step = step_.lock ();
    if (step)
        out_of_date = read1(step)->out_of_date();

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
        sleep(int sleep_ms) volatile
{
    Timer t;

    Step::Ptr pstep = ReadPtr(this)->step_.lock();
    Bedroom::Ptr bedroom = ReadPtr(this)->bedroom_.lock();

    if (!pstep || !bedroom)
        return false;

    Bedroom::Bed bed = bedroom->getBed();

    for (;;) {
        {
            Step::WritePtr step(pstep);

            if (!(ReadPtr(this)->needed_samples_ & step->out_of_date ()))
                return true;

            step->sleepWhileTasks (left(t, sleep_ms));

            if (!(ReadPtr(this)->needed_samples_ & step->out_of_date ()))
                return true;
        }

        bed.sleep(left(t, sleep_ms));

        if (0 <= sleep_ms && 0 == left(t, sleep_ms))
            return false;
    }
}


} // namespace Processing
} // namespace Signal

#include "dag.h"
#include "targets.h"

namespace Signal {
namespace Processing {

void TargetNeeds::
        test()
{
    // It should describe what needs to be computed for a target.
    {
        Bedroom::Ptr bedroom(new Bedroom);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));

        TargetNeeds::Ptr target_needs( new TargetNeeds(step, bedroom) );

        Signal::Intervals initial_valid(0,60);
        write1(step)->registerTask(0, initial_valid.spannedInterval ());

        EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( read1(step)->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( read1(target_needs)->out_of_date(), Signal::Interval() );
        write1(target_needs)->updateNeeds(Signal::Interval(-15,5));
        EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval::Interval_ALL );
        EXCEPTION_ASSERT_EQUALS( read1(step)->not_started(), ~initial_valid );
        EXCEPTION_ASSERT_EQUALS( read1(target_needs)->out_of_date(), Signal::Interval(-15,5) );
        EXCEPTION_ASSERT_EQUALS( read1(target_needs)->not_started(), Signal::Interval(-15,0) );
    }

    // It should not wakeup the bedroom if nothing has changed
    {
        // Note; this is more helpful to do a less noisy debugging than to increase any performance
        Bedroom::Ptr bedroom(new Bedroom);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        // Validate a bit Signal::Interval(0,10) of the step
        write1(step)->registerTask(0, Signal::Interval(0,10));
        write1(step)->finishTask(0, Signal::pBuffer(new Signal::Buffer(Signal::Interval(0,10),1,1)));

        TargetNeeds::Ptr target_needs( new TargetNeeds(step, bedroom) );

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,5));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(t.elapsed (), 1e-3); // Should not sleep since updateNeeds needs something new
        }

        {
            write1(target_needs)->updateNeeds(Signal::Interval(-15,5));

            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
            EXCEPTION_ASSERT_LESS(t.elapsed (), 3e-3); // Should not sleep for too long
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,4));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
            EXCEPTION_ASSERT_LESS(t.elapsed (), 3e-3); // Should not sleep for too long
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,6));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(t.elapsed (), 1e-3); // Should not sleep since updateNeeds needs something new
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,6),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX,
                                              Signal::Intervals(6,7) );
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
            EXCEPTION_ASSERT_LESS(t.elapsed (), 3e-3); // Should not sleep for too long
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,7));
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(t.elapsed (), 1e-3); // Should not sleep since updateNeeds needs something new
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,7),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX,
                                              Signal::Intervals(6,7) );
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(2e-3, t.elapsed ()); // Should sleep, updateNeeds didn't affect this
            EXCEPTION_ASSERT_LESS(t.elapsed (), 3e-3); // Should not sleep for too long
        }

        {
            Timer t;
            Bedroom::Bed bed = bedroom->getBed();
            write1(target_needs)->updateNeeds(Signal::Interval(-15,7),
                                              Signal::Interval::IntervalType_MIN,
                                              Signal::Interval::IntervalType_MAX,
                                              Signal::Intervals(5,7) );
            bed.sleep (2);
            EXCEPTION_ASSERT_LESS(t.elapsed (), 1e-3); // Should not sleep since updateNeeds needs something new
        }
    }
}

} // namespace Processing
} // namespace Signal
