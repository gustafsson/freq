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

        if (!worker) {
            // The worker has been deleted
            Signal::ComputingEngine::Ptr ce = i->first;
            dead[ce] = DeadEngines::mapped_type(0, "");

        } else if (!worker->isRunning ()) {
            // The worker has stopped but has not yet been deleted
            Signal::ComputingEngine::Ptr ce = i->first;
            dead[ce] = DeadEngines::mapped_type(worker->exception_type(), worker->exception_what());

            worker->deleteLater ();
        }
    }

    for (DeadEngines::iterator i=dead.begin (); i != dead.end(); ++i) {
        workers.erase (i->first);
    }

    if (!dead.empty ())
        updateWorkers();

    return dead;
}


class GetEmptyTaskMock: public GetTask {
public:
    GetEmptyTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        get_task_count++;
        throw std::logic_error("test crash");
        return Task::Ptr();
    }
};


void Schedule::
        test()
{
    // It should start and stop computing engines as they are added and removed
    {
//        GetDagTaskAlgorithm::Ptr algorithm(new ScheduleAlgorithm(workers_));
//        GetTask::Ptr get_dag_tasks(new GetDagTask(g, algorithm));
//        GetTask::Ptr wait_for_task(new ScheduleGetTask(get_dag_tasks));

        GetTask::Ptr gettaskp(new GetEmptyTaskMock);
        GetTask::WritePtr gettask(gettaskp);
        GetEmptyTaskMock* gettaskmock = dynamic_cast<GetEmptyTaskMock*>(&*gettask);
        Schedule schedule(gettaskp);

        int workers = 4;
        for (int i=0; i<workers; ++i)
            schedule.addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));

        usleep(6000);

        EXCEPTION_ASSERT_EQUALS(gettaskmock->get_task_count, workers);

        Schedule::DeadEngines dead = schedule.clean_dead_workers ();
        Engines engines = schedule.getWorkers();

        // If failing here, try to increase the sleep period above.
        EXCEPTION_ASSERT_EQUALS(engines.size (), 0);
        EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)workers);
    }
}


} // namespace Processing
} // namespace Signal
