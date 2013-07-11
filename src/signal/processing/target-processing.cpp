#include "target.h"

namespace Signal {
namespace Processing {

Target::
        Target(Step::Ptr step)
    :
      last_request_(boost::posix_time::min_date_time),
      work_center_(Signal::Interval::IntervalType_MIN),
      step_(step)
{
}


Step::Ptr Target::
        step() const
{
    return step_;
}


boost::posix_time::ptime Target::
        last_request() const
{
    return last_request_;
}


Signal::IntervalType Target::
        work_center() const
{
    return work_center_;
}


} // namespace Processing
} // namespace Signal
