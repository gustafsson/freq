#include "targetinvalidator.h"

#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;

namespace Signal {
namespace Processing {

TargetInvalidator::
        TargetInvalidator(Invalidator::Ptr invalidator, Target::Ptr target)
    :
      invalidator_(invalidator),
      target_(target)
{
}


void TargetInvalidator::
        update(int prio, Signal::IntervalType center, Signal::Intervals intervals)
{
    Target::WritePtr t(target_);
    invalidator_->deprecateCache (t->step, intervals);
    ptime now = microsec_clock::local_time();
    now += time_duration(0,0,prio);
    t->last_request = now;
    t->work_center = center;
}


void TargetInvalidator::
        test()
{

}


} // namespace Processing
} // namespace Signal
