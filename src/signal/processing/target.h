#ifndef SIGNAL_PROCESSING_TARGET_H
#define SIGNAL_PROCESSING_TARGET_H

#include "signal/intervals.h"
#include "step.h"

#include <boost/date_time/posix_time/ptime.hpp>

namespace Signal {
namespace Processing {

/**
 * @brief The TargetInfo class should provide information to prioritize tasks.
 *
 * Issues:
 * rename to TargetInfo.
 */
class Target: public VolatilePtr<Target>
{
public:
    Target(Step::Ptr step) : step(step) {}

    const Step::Ptr step;
    boost::posix_time::ptime last_request;
    Signal::IntervalType work_center;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGET_H
