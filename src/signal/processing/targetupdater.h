#ifndef SIGNAL_PROCESSING_TARGETUPDATER_H
#define SIGNAL_PROCESSING_TARGETUPDATER_H

#include "itargetupdater.h"
#include "iinvalidator.h"
#include "target.h"

namespace Signal {
namespace Processing {

class TargetUpdater: public ITargetUpdater
{
public:
    TargetUpdater(IInvalidator::Ptr invalidator, Target::Ptr target);

    void update(int prio, Signal::IntervalType center, Signal::Intervals intervals);

private:
    IInvalidator::Ptr invalidator_;
    Target::Ptr target_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETUPDATER_H
