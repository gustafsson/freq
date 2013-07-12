#ifndef SIGNAL_PROCESSING_TARGET_H
#define SIGNAL_PROCESSING_TARGET_H

#include "signal/intervals.h"
#include "step.h"

#include <boost/date_time/posix_time/ptime.hpp>

namespace Signal {
namespace Processing {

class Target: public VolatilePtr<Target>
{
public:
    Target(Step::Ptr step);

    Step::Ptr step() const;

    boost::posix_time::ptime last_request() const;
    Signal::IntervalType work_center() const;

    virtual Signal::Intervals out_of_date(Signal::Intervals skip = Signal::Intervals()) = 0;

protected:
    boost::posix_time::ptime last_request_;
    Signal::IntervalType work_center_;

private:
    Step::Ptr step_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGET_H
