#ifndef SIGNAL_PROCESSING_TARGET_H
#define SIGNAL_PROCESSING_TARGET_H

#include "signal/intervals.h"
#include "step.h"

namespace Signal {
namespace Processing {

class Target: public VolatilePtr<Target>
{
public:
    Target(Step::Ptr step);

    Step::Ptr step();

    boost::posix_time::ptime timestamp;

    virtual Signal::Intervals out_of_date(Signal::Intervals skip = Signal::Intervals()) = 0;

    int center;

private:
    Step::Ptr step_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGET_H
