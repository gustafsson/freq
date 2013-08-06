#include "targetneeds.h"
#include "step.h"
#include "bedroom.h"

#include "tools/support/timer.h"

#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetNeeds::
        TargetNeeds(Step::WeakPtr step, Bedroom::WeakPtr bedroom)
    :
      step_(step),
      bedroom_(bedroom)
{
    EXCEPTION_ASSERT(step.lock ());
    EXCEPTION_ASSERT(bedroom.lock ());
}


void TargetNeeds::
        updateNeeds(Signal::Intervals needed_samples, int prio, Signal::IntervalType center, Signal::Intervals invalidate)
{
    needed_samples_ = needed_samples;

    ptime now = microsec_clock::local_time();
    last_request_ = now + time_duration(0,0,prio);

    work_center_ = center;

    Step::Ptr step = step_.lock ();
    if (step && invalidate)
        write1(step)->deprecateCache(invalidate);

    Bedroom::Ptr bedroom = bedroom_.lock ();
    if (bedroom)
        bedroom->wakeup();
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


int left(const Tools::Support::Timer& t, int sleep_ms) {
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
    Tools::Support::Timer t;

    Step::Ptr pstep = ReadPtr(this)->step_.lock();
    Bedroom::Ptr bedroom = ReadPtr(this)->bedroom_.lock();

    if (!pstep || !bedroom)
        return false;

    for (;;) {
        bedroom->wakeup();

        {
            Step::WritePtr step(pstep);

            if (!(ReadPtr(this)->needed_samples_ & step->out_of_date ()))
                return true;

            step->sleepWhileTasks (left(t, sleep_ms));

            if (!(ReadPtr(this)->needed_samples_ & step->out_of_date ()))
                return true;
        }

        bedroom->sleep(left(t, sleep_ms));

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
    Bedroom::Ptr bedroom(new Bedroom);
    Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));

    TargetNeeds::Ptr target_needs( new TargetNeeds(step, bedroom) );

    Signal::Intervals initial_valid(0,60);
    write1(step)->registerTask(0, initial_valid.spannedInterval ());

    EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval::Interval_ALL );
    EXCEPTION_ASSERT_EQUALS( read1(step)->not_started(), ~initial_valid );
    EXCEPTION_ASSERT_EQUALS( read1(target_needs)->out_of_date(), Signal::Interval() );
    write1(target_needs)->updateNeeds(Signal::Interval(-15,5), 0, 0);
    EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval::Interval_ALL );
    EXCEPTION_ASSERT_EQUALS( read1(step)->not_started(), ~initial_valid );
    EXCEPTION_ASSERT_EQUALS( read1(target_needs)->out_of_date(), Signal::Interval(-15,5) );
    EXCEPTION_ASSERT_EQUALS( read1(target_needs)->not_started(), Signal::Interval(-15,0) );
}

} // namespace Processing
} // namespace Signal
