#ifndef SIGNAL_PROCESSING_TARGETS_H
#define SIGNAL_PROCESSING_TARGETS_H

#include "step.h"
#include "targetneeds.h"
#include "bedroom.h"
#include "dag.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Targets class should keep track of targets and let callers update
 * what each target needs afterwards.
 *
 * Multiple target can be added for the same Step.
 */
class Targets: public VolatilePtr<Targets>
{
public:
    typedef std::vector<TargetNeeds::Ptr> TargetNeedsCollection;

    Targets(Bedroom::Ptr bedroom);

    /**
     * @brief insert adds a new target to this collection of Targets.
     *
     * @param step A step for which to create a TargetNeed.
     *
     * @return A TargetNeeds instance to update what schedulers should focus
     * on in order to keep the target up-to-date. Note that the scheduler will
     * not create any task at all for this Target until you have specified
     * needed_samples through TargetNeeds::updateNeeds.
     *
     * Ownership is of TargetNeeds is given to the caller.
     */
    TargetNeeds::Ptr              addTarget(Step::Ptr step);
    TargetNeedsCollection         getTargets() const;

private:
    Bedroom::Ptr bedroom_;

    std::vector<TargetNeeds::WeakPtr> targets;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETS_H
