#ifndef SIGNAL_PROCESSING_TARGETS_H
#define SIGNAL_PROCESSING_TARGETS_H

#include "step.h"
#include "target.h"
#include "targetupdater.h"
#include "bedroom.h"
#include "dag.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Targets class should keep track of targets and let callers update
 * what each target needs afterwards.
 */
class Targets: public VolatilePtr<Targets>
{
public:
    Targets(Dag::Ptr dag, Bedroom::Ptr bedroom);

    ITargetUpdater::Ptr      addTarget(Step::Ptr step);
    void                     removeTarget(Step::Ptr step);
    std::vector<Step::Ptr>   getTargetSteps() const;
    std::vector<Target::Ptr> getTargets() const;

private:
    Dag::Ptr dag_;
    Bedroom::Ptr bedroom_;

    typedef std::vector<Target::Ptr> TargetInfos;
    TargetInfos targets;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETS_H
