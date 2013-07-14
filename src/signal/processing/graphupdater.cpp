#include <QObject>
#include "graphupdater.h"
#include "schedulegettask.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphUpdater::
        GraphUpdater(Dag::Ptr dag, WorkerBedroom::Ptr worker_bedroom)
    :
      dag_(dag),
      worker_bedroom_(worker_bedroom)
{
    EXCEPTION_ASSERT(dag);
    EXCEPTION_ASSERT(worker_bedroom_);
}


void GraphUpdater::
        deprecateCache(Step::Ptr s, Signal::Intervals /*what*/) const
{
    deprecateCache(Dag::ReadPtr(dag_), s);

    write1(worker_bedroom_)->wakeup ();
}


void GraphUpdater::
        deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const
{
    write1(s)->deprecateCache(Signal::Intervals::Intervals_ALL);

    BOOST_FOREACH(Step::Ptr ts, dag->targetSteps(s)) {
        deprecateCache(dag, ts);
    }
}

} // namespace Processing
} // namespace Signal
