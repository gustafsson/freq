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

    updateWorkers();

    // The computation is a background process with a priority one step lower than NormalPriority
    w->start (QThread::LowPriority);
}


void Schedule::
        removeComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    EngineWorkerMap::iterator worker = workers.find (ce);
    if (worker == workers.end ())
        EXCEPTION_ASSERTX(false, "No such engine");

    // Don't try to delete a running thread.
    worker->second->exit_nicely_and_delete();
    workers.erase (worker); // This doesn't delete worker, worker deletes itself (if there are any additional tasks).

    updateWorkers();
}


void Schedule::
        updateWorkers()
{
    std::vector<Signal::ComputingEngine::Ptr> engines;

    BOOST_FOREACH(EngineWorkerMap::value_type ewp, workers) {
        engines.push_back (ewp.first);
    }

    dynamic_cast<ScheduleWorkers*>((Workers*)write1(workers_))->workers_ = engines;
}


void Schedule::
        test()
{

}


} // namespace Processing
} // namespace Signal
