#include "targets.h"
#include "graphinvalidator.h"

namespace Signal {
namespace Processing {

Targets::
        Targets(Bedroom::Ptr bedroom)
    :
      bedroom_(bedroom)
{
}


TargetNeeds::Ptr Targets::
        addTarget(Step::Ptr step)
{
    TargetNeeds::Ptr target(new TargetNeeds(step, bedroom_));
    return target;
}


void Targets::
        removeTarget(Step::Ptr step)
{
    for (TargetInfos::iterator i=targets.begin (); i!=targets.end (); ++i) {
        TargetNeeds::Ptr t = *i;
        if (read1(t)->step() == step)
            targets.erase (i);
    }
}


std::vector<Step::Ptr> Targets::
        getTargetSteps() const
{
    std::vector<Step::Ptr> T;

    for (TargetInfos::const_iterator i=targets.begin (); i!=targets.end (); ++i) {
        TargetNeeds::Ptr t = *i;
        T.push_back (read1(t)->step());
    }

    return T;
}


std::vector<TargetNeeds::Ptr> Targets::
        getTargets() const
{
    return targets;
}

} // namespace Processing
} // namespace Signal



namespace Signal {
namespace Processing {

void Targets::
        test()
{
    // It should keep track of targets and let callers update what each target needs afterwards
    {
        // setup
        Bedroom::Ptr bedroom(new Bedroom);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));

        Targets::Ptr targets(new Targets(bedroom));
        TargetNeeds::Ptr updater = write1(targets)->addTarget(step);
        EXCEPTION_ASSERT(updater);
        EXCEPTION_ASSERT(read1(updater)->step() == step);
    }
}

} // namespace Processing
} // namespace Signal
