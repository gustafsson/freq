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
 */
class Targets: public VolatilePtr<Targets>
{
public:
    Targets(Bedroom::Ptr bedroom);

    TargetNeeds::Ptr              addTarget(Step::Ptr step);
    void                          removeTarget(Step::Ptr step);
    std::vector<Step::Ptr>        getTargetSteps() const;
    std::vector<TargetNeeds::Ptr> getTargets() const;

private:
    Bedroom::Ptr bedroom_;

    typedef std::vector<TargetNeeds::Ptr> TargetInfos;
    TargetInfos targets;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETS_H
