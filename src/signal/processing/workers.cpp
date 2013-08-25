// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "workers.h"
#include "targetschedule.h"
#include "timer.h"

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

namespace Signal {
namespace Processing {


Workers::
        Workers(ISchedule::Ptr schedule)
    :
      schedule_(schedule)
{
}


Workers::
        ~Workers()
{
    schedule_.reset ();

    terminate_workers ();

    print (clean_dead_workers());
}


Worker::Ptr Workers::
        addComputingEngine(Signal::ComputingEngine::Ptr ce)
{
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
    EngineWorkerMap::iterator worker = workers_map_.find (ce);
    if (worker == workers_map_.end ())
        EXCEPTION_ASSERTX(false, "No such engine");

    // Don't try to delete a running thread.
    if (worker->second)
        worker->second->exit_nicely_and_delete();

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
            dead[ce] = boost::exception_ptr();

        } else {
            // Worker::delete_later() is carried out in the event loop of the
            // thread in which it was created. If the current thread here is
            // the same thread, we know that worker is not deleted between the
            // !worker check about and the usage of worker below:
            if (!worker->isRunning ()) {
                // The worker has stopped but has not yet been deleted
                Signal::ComputingEngine::Ptr ce = i->first;
                dead[ce] = worker->caught_exception ();

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


void Workers::
        rethrow_worker_exception()
{
    for(EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        QPointer<Worker> worker = i->second;

        if (worker && worker->caught_exception ()) {
            workers_map_.erase (i);
            rethrow_exception(worker->caught_exception ());
        }
    }
}


bool Workers::
        terminate_workers(int timeout)
{
    TIME_TERMINATE TaskTimer ti("terminate_workers");

    remove_all_engines(timeout);

    bool ok = true;

    TIME_TERMINATE TaskInfo("terminate");
    BOOST_FOREACH(EngineWorkerMap::value_type i, workers_map_) {
        if (i.second) i.second->terminate ();
    }

    TIME_TERMINATE TaskInfo("wait(1000)");
    // Wait for terminate to take effect
    BOOST_FOREACH(EngineWorkerMap::value_type i, workers_map_) {
        if (i.second) ok &= i.second->wait (1000);
    }

    return ok;
}


bool Workers::
        remove_all_engines(int timeout) const
{
    TIME_TERMINATE TaskTimer ti("remove_all_engines");

    bool ok = true;

    // Make sure the workers doesn't start anything new
    BOOST_FOREACH(EngineWorkerMap::value_type i, workers_map_) {
        if (i.second) i.second->exit_nicely_and_delete();
    }

    TIME_TERMINATE TaskInfo("wait(%d)", timeout);
    // Give ISchedule and Task 1.0 s to return.
    BOOST_FOREACH(EngineWorkerMap::value_type i, workers_map_) {
        if (i.second) ok &= i.second->wait (timeout);
    }

    return ok;
}


void Workers::
        print(const DeadEngines& engines)
{
    if (engines.empty ())
        return;

    TaskInfo ti("Dead engines");

    BOOST_FOREACH(Workers::DeadEngines::value_type e, engines) {
        Signal::ComputingEngine::Ptr engine = e.first;
        boost::exception_ptr x = e.second;

        if (x)
            TaskInfo(boost::format("engine %1% failed.\n%2%")
                     % (engine ? vartype(*engine.get ()) : (vartype(engine.get ())+"==0"))
                     % boost::diagnostic_information(x));
        else
            TaskInfo(boost::format("engine %1% stopped")
                     % (engine ? vartype(*engine.get ()) : (vartype(engine.get ())+"==0")));
    }
}


class GetEmptyTaskMock: public ISchedule {
public:
    GetEmptyTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        WritePtr w(this);
        GetEmptyTaskMock* self = dynamic_cast<GetEmptyTaskMock*>(&*w);

        //GetEmptyTaskMock* self = const_cast<GetEmptyTaskMock*>(this);
        //__sync_fetch_and_add (&self->get_task_count, 1);

        // could also use boost::detail::atomic_count get_task_count
        self->get_task_count++;
        if (self->get_task_count%2)
            throw std::logic_error("test crash");
        else
            return Task::Ptr();
    }
};


class BlockScheduleMock: public ISchedule, public Bedroom {
protected:
    virtual Task::Ptr getTask() volatile {
        wakeup ();

        // This should block the thread and be aborted by QThread::terminate
        dont_return();

        EXCEPTION_ASSERT( false );

        return Task::Ptr();
    }

    virtual void dont_return() volatile = 0;
};

class SleepScheduleMock: public BlockScheduleMock {
    virtual void dont_return() volatile {
        // Don't use this->sleep() as that semaphore is used for something else.
        Bedroom().sleep ();
    }
};


class LockScheduleMock: public BlockScheduleMock {
    virtual void dont_return() volatile {
        ISchedule::readWriteLock()->lockForWrite (); // ok
        ISchedule::readWriteLock()->lockForWrite (); // lock
    }
};


class BusyScheduleMock: public BlockScheduleMock {
    virtual void dont_return() volatile {
        for(;;) {
            usleep(0); // Allow OS scheduling to kill the thread ("for(;;);" would not)
        }
    }
};


void Workers::
        test()
{
    // It should start and stop computing engines as they are added and removed
    double maxwait = 0;
    for (int j=0;j<100; j++){
        ISchedule::Ptr schedule(new GetEmptyTaskMock);
        Workers workers(schedule);
        workers.rethrow_worker_exception(); // Should do nothing

        Timer t;
        int worker_count = 40; // Multiplying by 10 multiplies the elapsed time by a factor of 100.
        std::list<Worker::Ptr> workerlist;
        Worker::Ptr w = workers.addComputingEngine(Signal::ComputingEngine::Ptr());
        workerlist.push_back (w);
        for (int i=1; i<worker_count; ++i) {
            Worker::Ptr w = workers.addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
            workerlist.push_back (w);
        }

        // Wait until they're done
        BOOST_FOREACH (Worker::Ptr& w, workerlist) w->wait ();
        maxwait = std::max(maxwait, t.elapsed ());

        int get_task_count = ((const GetEmptyTaskMock*)&*read1(schedule))->get_task_count;
        EXCEPTION_ASSERT_EQUALS(get_task_count, worker_count);

        // It should forward exceptions from workers
        try {
            workers.rethrow_worker_exception();
            EXCEPTION_ASSERTX(false, "Expected exception");
        } catch (const std::exception&) {}

        Workers::DeadEngines dead = workers.clean_dead_workers ();
        Engines engines = workers.workers();

        EXCEPTION_ASSERT_EQUALS(engines.size (), 0);
        EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)worker_count-1); // One was cleared by catching its exception above

        // When dead workers are cleared there should not be any exceptions thrown
        workers.rethrow_worker_exception();
    }

    EXCEPTION_ASSERT_LESS(maxwait, 0.02);

    // It should terminate all threads when it's closed
    {
        ISchedule::Ptr schedule[] = {
            ISchedule::Ptr(new SleepScheduleMock),
            ISchedule::Ptr(new LockScheduleMock),
            ISchedule::Ptr(new BusyScheduleMock)
        };

        for (unsigned i=0; i<sizeof(schedule)/sizeof(schedule[0]); i++) {
            Timer t;
            {
                ISchedule::Ptr s = schedule[i];
                TaskInfo ti(boost::format("%s") % vartype(*s));

                Workers workers(s);

                workers.addComputingEngine(Signal::ComputingEngine::Ptr());

                // Wait until the schedule has been called
                dynamic_cast<volatile Bedroom*>(s.get ())->sleep ();

                workers.terminate_workers (10);
                workers.rethrow_worker_exception ();
                workers.clean_dead_workers ();
            }
            float elapsed = t.elapsed ();
            float n = (i+1)*0.00001;
            EXCEPTION_ASSERT_LESS(0.01+n, elapsed);
            EXCEPTION_ASSERT_LESS(elapsed, 0.012+n);
        }
    }
}


} // namespace Processing
} // namespace Signal
