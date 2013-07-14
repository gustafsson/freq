#include "targets.h"
#include "graphupdater.h"
#include "targetinvalidator.h"

namespace Signal {
namespace Processing {

Targets::
        Targets(Dag::Ptr dag, WorkerBedroom::Ptr worker_bedroom)
    :
      dag_(dag),
      worker_bedroom_(worker_bedroom)
{
}


TargetUpdater::Ptr Targets::
        addTarget(Step::Ptr step)
{
    Invalidator::Ptr invalidator(new GraphUpdater(dag_, worker_bedroom_ ));
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
        getTargets() const
{
    std::vector<Step::Ptr> T;

    for (TargetInfos::const_iterator i=targets.begin (); i!=targets.end (); ++i) {
        Target::Ptr t = *i;
        T.push_back (read1(t)->step);
    }

    return T;
}

} // namespace Processing
} // namespace Signal
