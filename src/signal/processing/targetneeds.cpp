#include "targetneeds.h"
#include "step.h"

#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetNeeds::
        TargetNeeds(boost::shared_ptr<volatile Step> step, Bedroom::Ptr bedroom)
    :
      step_(step),
      bedroom(bedroom)
{
    EXCEPTION_ASSERT(step);
    EXCEPTION_ASSERT(bedroom);
}


void TargetNeeds::
        updateNeeds(Signal::Intervals intervals, int prio, Signal::IntervalType center)
{
    write1(step_)->setInvalid(intervals);

    ptime now = microsec_clock::local_time();
    last_request_ = now + time_duration(0,0,prio);

    work_center_ = center;

    bedroom->wakeup();
}


const boost::shared_ptr<volatile Step> TargetNeeds::
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
    Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));

    TargetNeeds::Ptr updater( new TargetNeeds(step, bedroom) );

    write1(step)->setInvalid(Signal::Interval(8,9));
    EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval(8,9) );
    write1(updater)->updateNeeds(Signal::Interval(4,5), 0, 0);
    EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Interval(4,5) );
}

} // namespace Processing
} // namespace Signal
