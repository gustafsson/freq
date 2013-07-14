#include "targets.h"
#include "graphinvalidator.h"
#include "itargetupdater.h"

namespace Signal {
namespace Processing {

Targets::
        Targets(Dag::Ptr dag, Bedroom::Ptr bedroom)
    :
      dag_(dag),
      bedroom_(bedroom)
{
}


ITargetUpdater::Ptr Targets::
        addTarget(Step::Ptr step)
{
    IInvalidator::Ptr invalidator(new GraphInvalidator(dag_, bedroom_ ));
    Target::Ptr target(new Target(step));
    ITargetUpdater::Ptr target_updater(new TargetUpdater(invalidator, target ));

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



namespace Signal {
namespace Processing {

void Targets::
        test()
{
    // It should keep track of targets and let callers update what each target needs afterwards
    {
        // setup
        Dag::Ptr dag(new Dag);
        Bedroom::Ptr bedroom(new Bedroom);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));
        write1(step)->setInvalid(Signal::Intervals());
        write1(dag)->appendStep(step);


        Targets::Ptr targets(new Targets(dag, bedroom));
        ITargetUpdater::Ptr updater = write1(targets)->addTarget(step);
        Signal::Intervals whatever;

        EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Intervals() );
        updater->update(0, 0, whatever);
        EXCEPTION_ASSERT_EQUALS( read1(step)->out_of_date(), Signal::Intervals::Intervals_ALL );
    }
}

} // namespace Processing
} // namespace Signal
