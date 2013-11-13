#include "targets.h"
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Targets::
        Targets(Bedroom::WeakPtr bedroom)
    :
      bedroom_(bedroom)
{
}


TargetNeeds::Ptr Targets::
        addTarget(Step::WeakPtr step)
{
    TargetNeeds::Ptr target(new TargetNeeds(step, bedroom_));
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

    BOOST_FOREACH(const TargetNeeds::WeakPtr& i, targets) {
        TargetNeeds::Ptr t = i.lock ();
        if (t)
            C.push_back (t);
    }

    return C;
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
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));

        Targets::Ptr targets(new Targets(bedroom));
        TargetNeeds::Ptr updater( write1(targets)->addTarget(step) );
        EXCEPTION_ASSERT(updater);
        EXCEPTION_ASSERT(read1(updater)->step().lock() == step);
    }
}

} // namespace Processing
} // namespace Signal
