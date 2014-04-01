#include "targets.h"
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Targets::
        Targets(INotifier::weak_ptr notifier)
    :
      notifier_(notifier)
{
}


TargetNeeds::Ptr Targets::
        addTarget(Step::Ptr::weak_ptr step)
{
    TargetNeeds::Ptr target(new TargetNeeds(step, notifier_));
    targets.push_back (target);

    // perform gc
    Targets::TargetNeedsCollection T = getTargets();
    targets.clear ();
    targets.reserve (T.size ());
    BOOST_FOREACH(const TargetNeeds::Ptr& i, T) {
        targets.push_back (i);
    }

    return target;
}


Targets::TargetNeedsCollection Targets::
        getTargets() const
{
    TargetNeedsCollection C;
    C.reserve (targets.size ());

    BOOST_FOREACH(const TargetNeeds::Ptr::weak_ptr& i, targets) {
        TargetNeeds::Ptr t = i.lock ();
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
        Bedroom::Ptr bedroom(new Bedroom);
        BedroomNotifier::Ptr bedroom_notifier(new BedroomNotifier(bedroom));
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));

        Targets::Ptr targets(new Targets(bedroom_notifier));
        TargetNeeds::Ptr updater( targets.write ()->addTarget(step) );
        EXCEPTION_ASSERT(updater);
        EXCEPTION_ASSERT(updater.read ()->step().lock() == step);
    }
}

} // namespace Processing
} // namespace Signal
