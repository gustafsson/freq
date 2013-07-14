#include "targets.h"
#include "graphinvalidator.h"
#include "targetinvalidator.h"

namespace Signal {
namespace Processing {

Targets::
        Targets(Dag::Ptr dag, Bedroom::Ptr bedroom)
    :
      dag_(dag),
      bedroom_(bedroom)
{
}


TargetUpdater::Ptr Targets::
        addTarget(Step::Ptr step)
{
    Invalidator::Ptr invalidator(new GraphInvalidator(dag_, bedroom_ ));
    Target::Ptr target(new Target(step));
    TargetUpdater::Ptr target_updater(new TargetInvalidator(invalidator, target ));

    return target_updater;
}


void Targets::
        removeTarget(Step::Ptr step)
{
    for (TargetInfos::iterator i=targets.begin (); i!=targets.end (); ++i) {
        Target::Ptr t = *i;
        if (read1(t)->step == step)
            targets.erase (i);
    }
}


std::vector<Step::Ptr> Targets::
        getTargetSteps() const
{
    std::vector<Step::Ptr> T;

    for (TargetInfos::const_iterator i=targets.begin (); i!=targets.end (); ++i) {
        Target::Ptr t = *i;
        T.push_back (read1(t)->step);
    }

    return T;
}


std::vector<Target::Ptr> Targets::
        getTargets() const
{
    return targets;
}

} // namespace Processing
} // namespace Signal
