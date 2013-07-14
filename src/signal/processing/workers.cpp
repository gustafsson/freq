// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "workers.h"

#include "targetschedule.h"

namespace Signal {
namespace Processing {


Workers::
        Workers(ISchedule::Ptr schedule)
    :
      schedule_(schedule)
{
}


void Workers::
        addComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    if (workers_map_.find (ce) != workers_map_.end ())
        EXCEPTION_ASSERTX(false, "Engine already added");

    Worker::Ptr w(new Worker(ce, schedule_));
    workers_map_[ce] = w;

    updateWorkers();

    // The computation is a background process with a priority one step lower than NormalPriority
    w->start (QThread::LowPriority);
}


void Workers::
        removeComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    EngineWorkerMap::iterator worker = workers_map_.find (ce);
    if (worker == workers_map_.end ())
        EXCEPTION_ASSERTX(false, "No such engine");

    // Don't try to delete a running thread.
    worker->second->exit_nicely_and_delete();
    workers_map_.erase (worker); // This doesn't delete worker, worker deletes itself (if there are any additional tasks).

    updateWorkers();
}


const Workers::Engines& Workers::
        workers() const
{
    return workers_;
}


size_t Workers::
        n_workers() const
{
    return workers_.size();
}


void Workers::
        updateWorkers()
{
    Engines engines;

    BOOST_FOREACH(EngineWorkerMap::value_type ewp, workers_map_) {
        engines.push_back (ewp.first);
    }

    workers_ = engines;
}


Workers::DeadEngines Workers::
        clean_dead_workers()
{
    DeadEngines dead;

    for(EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
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
        workers_map_.erase (i->first);
    }

    if (!dead.empty ())
        updateWorkers();

    return dead;
}


class GetEmptyTaskMock: public ISchedule {
public:
    GetEmptyTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        get_task_count++;
        if (get_task_count%2)
            throw std::logic_error("test crash");
        else
            return Task::Ptr();
    }
};


void Workers::
        test()
{
    // It should start and stop computing engines as they are added and removed
    {
        ISchedule::Ptr gettaskp(new GetEmptyTaskMock);
        ISchedule::WritePtr gettask(gettaskp);
        GetEmptyTaskMock* schedulemock = dynamic_cast<GetEmptyTaskMock*>(&*gettask);
        Workers schedule(gettaskp);

        int workers = 4;
        for (int i=0; i<workers; ++i)
            schedule.addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));

        usleep(12000);

        EXCEPTION_ASSERT_EQUALS(schedulemock->get_task_count, workers);

        Workers::DeadEngines dead = schedule.clean_dead_workers ();
        Engines engines = schedule.workers();

        // If failing here, try to increase the sleep period above.
        EXCEPTION_ASSERT_EQUALS(engines.size (), 0);
        EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)workers);
    }
}


} // namespace Processing
} // namespace Signal
