#include <QObject>
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphInvalidator::
        GraphInvalidator(Dag::Ptr dag, Bedroom::Ptr bedroom)
    :
      dag_(dag),
      bedroom_(bedroom)
{
    EXCEPTION_ASSERT(dag);
    EXCEPTION_ASSERT(bedroom_);
}


void GraphInvalidator::
        deprecateCache(Step::Ptr s, Signal::Intervals /*what*/) const
{
    deprecateCache(Dag::ReadPtr(dag_), s);

    bedroom_->wakeup ();
}


void GraphInvalidator::
        deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const
{
    write1(s)->deprecateCache(Signal::Intervals::Intervals_ALL);

    BOOST_FOREACH(Step::Ptr ts, dag->targetSteps(s)) {
        deprecateCache(dag, ts);
    }
}

} // namespace Processing
} // namespace Signal
