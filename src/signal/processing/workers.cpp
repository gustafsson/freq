// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "workers.h"
#include "targetschedule.h"
#include "tools/support/timer.h"

namespace Signal {
namespace Processing {


Workers::
        Workers(ISchedule::Ptr schedule)
    :
      schedule_(schedule)
{
}


Worker::Ptr Workers::
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

    return w;
}


void Workers::
        removeComputingEngine(Signal::ComputingEngine::Ptr ce)
{
    EXCEPTION_ASSERT(ce);

    EngineWorkerMap::iterator worker = workers_map_.find (ce);
    if (worker == workers_map_.end ())
        EXCEPTION_ASSERTX(false, "No such engine");

    // Don't try to delete a running thread.
    if (worker->second)
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

        } else {
            // Worker::delete_later() is carried out in the event loop of the
            // thread in which it was created. If the current thread here is
            // the same thread, we know that worker is not deleted between the
            // !worker check about and the usage of worker below:
            if (!worker->isRunning ()) {
                // The worker has stopped but has not yet been deleted
                Signal::ComputingEngine::Ptr ce = i->first;
                dead[ce] = DeadEngines::mapped_type(worker->exception_type(), worker->exception_what());

                worker->deleteLater ();
            }
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
        WritePtr w(this);
        GetEmptyTaskMock* self = dynamic_cast<GetEmptyTaskMock*>(&*w);

        self->get_task_count++;
        if (self->get_task_count%2)
            throw std::logic_error("test crash");
        else
            return Task::Ptr();
    }
};


void Workers::
        test()
{
    // It should start and stop computing engines as they are added and removed
    double maxwait = 0;
    for (int j=0; j<100; j++) {
        ISchedule::Ptr schedule(new GetEmptyTaskMock);
        Workers workers(schedule);

        int worker_count = 40;
        std::list<Worker::Ptr> workerlist;
        for (int i=0; i<worker_count; ++i) {
            Worker::Ptr w = workers.addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
            workerlist.push_back (w);
        }

        // Wait until they're done
        Tools::Support::Timer t;
        BOOST_FOREACH (Worker::Ptr& w, workerlist) w->wait (1);
        maxwait = std::max(maxwait, t.elapsed ());

        int get_task_count = dynamic_cast<const GetEmptyTaskMock*>(&*read1(schedule))->get_task_count;
        EXCEPTION_ASSERT_EQUALS(get_task_count, worker_count);

        Workers::DeadEngines dead = workers.clean_dead_workers ();
        Engines engines = workers.workers();

        // If failing here, try to increase the sleep period above.
        EXCEPTION_ASSERT_EQUALS(engines.size (), 0);
        EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)worker_count);
    }

    EXCEPTION_ASSERT_LESS(maxwait, 0.0005);
}


} // namespace Processing
} // namespace Signal
