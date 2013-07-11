// Include Boost.Foreach before any Qt includes to prevent conflicts with Qt foreach
#include <boost/foreach.hpp>

#include "schedule.h"

#include "schedulegettask.h"

namespace Signal {
namespace Processing {


Schedule::
        Schedule(Dag::Ptr g)
    :
      get_task(new ScheduleGetTask(g))
{
}


void Schedule::
        addComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    if (workers.find (ce) != workers.end ())
        EXCEPTION_ASSERTX(false, "Engine already added");

    Worker::Ptr w(new Worker(ce, get_task));
    workers[ce] = w;

    // The computation is a background process with a priority one step lower than NormalPriority
    w->start (QThread::LowPriority);
}


void Schedule::
        removeComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    if (workers.find (ce) == workers.end ())
        EXCEPTION_ASSERTX(false, "No such engine");

    // Don't try to delete a running thread.
    workers.erase (ce);
}


std::vector<Signal::ComputingEngine::Ptr> Schedule::
        getComputingEngines() const
{
    std::vector<Signal::ComputingEngine::Ptr> engines;

    BOOST_FOREACH(EngineWorkerMap::value_type ewp, workers) {
        engines.push_back (ewp.first);
    }

    return engines;
}


void Schedule::
        test()
{

}


} // namespace Processing
} // namespace Signal
