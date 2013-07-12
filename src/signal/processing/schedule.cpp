// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "schedule.h"

#include "schedulegettask.h"
#include "getdagtask.h"
#include "schedulealgorithm.h"

namespace Signal {
namespace Processing {


Schedule::
        Schedule(GetTask::Ptr get_task)
    :
      get_task(get_task)
{
/*    GetDagTaskAlgorithm::Ptr algorithm(new ScheduleAlgorithm(workers_));
    GetTask::Ptr get_dag_tasks(new GetDagTask(g, algorithm));
    GetTask::Ptr wait_for_task(new ScheduleGetTask(get_dag_tasks));
*/
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


const Schedule::Engines& Schedule::
        getWorkers() const
{
    return workers_;
}


void Schedule::
        updateWorkers()
{
    Engines engines;

    BOOST_FOREACH(EngineWorkerMap::value_type ewp, workers) {
        engines.push_back (ewp.first);
    }

    workers_ = engines;
}


Schedule::DeadEngines Schedule::
        clean_dead_workers()
{
    DeadEngines dead;

    for(EngineWorkerMap::iterator i=workers.begin (); i != workers.end(); ++i) {
        QPointer<Worker> worker = i->second;
        EXCEPTION_ASSERT (!worker); // This indicates an error in scheduling if a null task was returned

        if (!worker->isRunning ()) // This
        {
            // Do something intelligent
            Signal::ComputingEngine::Ptr ce = i->first;
            dead[ce] = DeadEngines::mapped_type(worker->exception_type(), worker->exception_what());

            workers.erase (i);
            i = workers.begin ();
        }
    }

    return dead;
}


void Schedule::
        test()
{

}


} // namespace Processing
} // namespace Signal
