#include "targets.h"
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Targets::
        Targets(INotifier::weak_ptr notifier)
    :
      state_(new State),
      notifier_(notifier)
{
}


TargetNeeds::ptr Targets::
        addTarget(Step::ptr::weak_ptr step)
{
    TargetNeeds::ptr target(new TargetNeeds(step.lock (), notifier_));

    auto state = state_.write ();
    State::Targets& targets = state->targets;
    targets.push_back (target);

    // perform gc
    Targets::TargetNeedsCollection T = getTargets(*state.get ());
    targets.clear ();
    targets.reserve (T.size ());
    BOOST_FOREACH(const TargetNeeds::ptr& i, T) {
        targets.push_back (i);
    }

    return target;
}


Targets::TargetNeedsCollection Targets::
        getTargets() const
{
    return getTargets(*state_.read ().get ());
}


Targets::TargetNeedsCollection Targets::
        getTargets(const State& state) const
{
    TargetNeedsCollection C;
    const State::Targets& targets = state.targets;
    C.reserve (targets.size ());

    for (const std::weak_ptr<TargetNeeds>& i: targets) {
        TargetNeeds::ptr t = i.lock ();
        if (t)
            C.push_back (t);
    }

    return C;
}


} // namespace Processing
} // namespace Signal

#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

void Targets::
        test()
{
    // It should keep track of targets and let callers update what each target needs afterwards
    {
        // setup
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr bedroom_notifier(new BedroomNotifier(bedroom));
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));

        Targets::ptr targets(new Targets(bedroom_notifier));
        TargetNeeds::ptr updater( targets->addTarget(step) );
        EXCEPTION_ASSERT(updater);
        EXCEPTION_ASSERT(updater->step().lock() == step);
    }
}

} // namespace Processing
} // namespace Signal
