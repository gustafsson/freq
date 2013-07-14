#ifndef SIGNAL_PROCESSING_TARGETINVALIDATOR_H
#define SIGNAL_PROCESSING_TARGETINVALIDATOR_H

#include "targetupdater.h"
#include "invalidator.h"
#include "dag.h"
#include "target.h"

namespace Signal {
namespace Processing {

class TargetInvalidator: public TargetUpdater
{
public:
    TargetInvalidator(Invalidator::Ptr invalidator, Target::Ptr target);

    virtual void update(int prio, Signal::IntervalType center, Signal::Intervals intervals);

private:
    Invalidator::Ptr invalidator_;
    Target::Ptr target_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETINVALIDATOR_H
