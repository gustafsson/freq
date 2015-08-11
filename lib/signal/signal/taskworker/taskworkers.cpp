#include "taskworkers.h"
#include "taskworker.h"

#include "thread_pool.h"
#include "signal/computingengine.h"
#include "demangle.h"
#include "expectexception.h"
#include "tasktimer.h"
#include "log.h"

#include <map>
#include <thread>

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

#define UNITTEST_STEPS
//#define UNITTEST_STEPS if(0)

using namespace std;
using namespace JustMisc;
using namespace Signal::Processing;

namespace Signal {
namespace TaskWorker {

TaskWorkers::TaskWorkers(ISchedule::ptr schedule, Bedroom::ptr bedroom)
    : Workers(schedule,bedroom)
{
}


void TaskWorkers::
        addComputingEngine(Signal::ComputingEngine::ptr ce)
{
    if (workers_map_.find (ce) != workers_map_.end ())
        EXCEPTION_ASSERTX(false, "Engine already added");

    TaskWorker::ptr w(new TaskWorker(ce, bedroom_, schedule_));
    workers_map_[ce] = std::move(w);

    updateWorkers();
}


void TaskWorkers::
        removeComputingEngine(Signal::ComputingEngine::ptr ce)
{
    const EngineWorkerMap::iterator worker = workers_map_.find (ce);
    if (worker != workers_map_.end ())
      {
        // Don't try to delete a running thread.
        if (worker->second && worker->second->isRunning())
          {
            worker->second->abort();
          }
        else
          {
            workers_map_.erase (worker);
            updateWorkers();
          }
      }
}


const Workers::Engines &TaskWorkers::workers() const
{
    return workers_;
}


size_t TaskWorkers::
        n_workers() const
{
    size_t N = 0;

    for(const auto& i : workers_map_) {
        const TaskWorker::ptr& worker = i.second;

        if (worker && worker->isRunning ())
            N++;
    }

    return N;
}


const Workers::EngineWorkerMap& TaskWorkers::
        workers_map() const
{
    return workers_map_;
}


TaskWorkers::DeadEngines TaskWorkers::
        clean_dead_workers()
{
    DeadEngines dead;

    for (EngineWorkerMap::iterator i=workers_map_.begin ();
         i != workers_map_.end();
         ++i)
    {
        const TaskWorker::ptr& worker = i->second;

        if (!worker) {
            // The worker has been deleted
            Signal::ComputingEngine::ptr ce = i->first;
            dead[ce] = std::exception_ptr();

        } else {
            // Worker::delete_later() is carried out in the event loop of the
            // thread in which it was created. If the current thread here is
            // the same thread, we know that worker is not deleted between the
            // !worker check about and the usage of worker below:
            if (!worker->isRunning ()) {
                // The worker has stopped but has not yet been deleted
                Signal::ComputingEngine::ptr ce = i->first;
                std::exception_ptr e = worker->caught_exception ();
                if (e) {
                    // Append engine typename info
                    // Make a boost exception if it wasn't
                    try {
                        std::rethrow_exception(e);
                    } catch (...) {
                        try {
                            boost::rethrow_exception (boost::current_exception ());
                        } catch (boost::exception& x) {
                            x << crashed_engine(ce) << crashed_engine_typename(ce?vartype(*ce):"(null)");
                            e = std::current_exception ();
                        }
                    }
                }
                dead[ce] = e;
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


void TaskWorkers::
        rethrow_any_worker_exception()
{
    for (EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        const TaskWorker::ptr& worker = i->second;

        if (worker) {
            std::exception_ptr e = worker->caught_exception ();
            if (e) {
                Signal::ComputingEngine::ptr ce = i->first;

                workers_map_.erase (i);

                try {
                    std::rethrow_exception(e);
                } catch (...) {
                    // Append engine typename info
                    // Make a boost exception if it wasn't
                    try {
                        boost::rethrow_exception (boost::current_exception ());
                    } catch (boost::exception& x) {
                        x << crashed_engine(ce) << crashed_engine_typename(ce?vartype(*ce):"(null)");
                        throw;
                    }
                }
            }
        }
    }
}


bool TaskWorkers::
        remove_all_engines(int timeout) const
{
    TIME_TERMINATE TaskTimer ti("remove_all_engines");

    // Make sure the workers doesn't start anything new
    for (const auto& i: workers_map_) {
        if (i.second) i.second->abort();
    }

    return wait(timeout);
}


bool TaskWorkers::
        wait(int timeout) const
{
    TIME_TERMINATE TaskTimer ti("wait(%d)", timeout);
    bool ok = true;

    for (const auto& i: workers_map_) {
        if (i.second)
            ok &= i.second->wait (timeout);
    }

    return ok;
}


void TaskWorkers::
        updateWorkers()
{
    Engines engines;

    for (const EngineWorkerMap::value_type& ewp: workers_map_)
        engines.push_back (ewp.first);

    workers_ = move(engines);
}

} // namespace TaskWorker
} // namespace Signal


namespace Signal {
namespace TaskWorker {

class GetEmptyTaskMock: public ISchedule {
public:
    GetEmptyTaskMock() : get_task_count(0) {}

    mutable std::atomic<int> get_task_count;

    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        //__sync_fetch_and_add (&self->get_task_count, 1);

        // could also use 'boost::detail::atomic_count get_task_count;'
        int v = get_task_count.fetch_add(1);
        if (v%2)
            throw std::logic_error("test crash");
        else
            return Task();
    }
};


class BlockScheduleMock: public ISchedule {
protected:
    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        Log("BlockScheduleMock wakeup");

        bedroom.wakeup ();

        // This should block the thread and be aborted by QThread::terminate
        dont_return();

        Log("BlockScheduleMock exception");

        EXCEPTION_ASSERT( false );

        Log("BlockScheduleMock return empty task");

        return Task();
    }

    virtual void dont_return() const = 0;

public:
    mutable Bedroom bedroom;
};

class SleepScheduleMock: public BlockScheduleMock {
    virtual void dont_return() const override {
        // Don't use this->getBed() as that Bedroom is used for something else.
        Bedroom().getBed().sleep ();
    }
};


class LockScheduleMock: public BlockScheduleMock {
    virtual void dont_return() const override {
        shared_state_mutex m;
        m.lock (); // ok
        m.lock (); // lock
    }
};


class BusyScheduleMock: public BlockScheduleMock {
    virtual void dont_return() const override {
        for(;;) usleep(0); // Allow OS scheduling to kill the thread (just "for(;;);" would not)
    }
};


void TaskWorkers::
        test()
{
    {
        UNITTEST_STEPS TaskInfo("It should start and stop computing engines as they are added and removed");

        // It should start and stop computing engines as they are added and removed
        double maxwait = 0;
        for (int j=0;j<100; j++)
        {
            ISchedule::ptr schedule(new GetEmptyTaskMock);
            Bedroom::ptr bedroom(new Bedroom);
            TaskWorkers workers(schedule, bedroom);
            workers.rethrow_any_worker_exception(); // Should do nothing

            Timer t;
            int worker_count = 40; // Number of threads to start. Multiplying by 10 multiplies the elapsed time by a factor of 100.
            workers.addComputingEngine(Signal::ComputingEngine::ptr());
            for (int i=1; i<worker_count; ++i)
                workers.addComputingEngine(Signal::ComputingEngine::ptr(new Signal::ComputingCpu));

            const EngineWorkerMap& workers_map = workers.workers_map_;
            std::list<Worker*> workerlist;
            for(const EngineWorkerMap::value_type& v : workers_map)
                workerlist.push_back (v.second.get());

            // Wait until they're done
            for (Worker* w: workerlist) { w->abort (); w->wait (); }
            maxwait = std::max(maxwait, t.elapsed ());

            int get_task_count = ((const GetEmptyTaskMock*)schedule.get ())->get_task_count;
            EXCEPTION_ASSERT_EQUALS(workerlist.size (), (size_t)worker_count);
            EXCEPTION_ASSERT_LESS_OR_EQUAL(worker_count, get_task_count);

            // It should forward exceptions from workers
            try {
                workers.rethrow_any_worker_exception();
                EXCEPTION_ASSERTX(false, "Expected exception");
            } catch (const std::exception& x) {
                const Signal::ComputingEngine::ptr* ce =
                        boost::get_error_info<crashed_engine>(x);
                EXCEPTION_ASSERT(ce);

                const std::string* cename =
                        boost::get_error_info<crashed_engine_typename>(x);
                EXCEPTION_ASSERT(cename);
            }

            TaskWorkers::DeadEngines dead = workers.clean_dead_workers ();
            Engines engines = workers.workers();

            EXCEPTION_ASSERT_EQUALS(engines.size (), 0u);
            EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)worker_count-1); // One was cleared by catching its exception above

            // When dead workers are cleared there should not be any exceptions thrown
            workers.rethrow_any_worker_exception();
        }

        EXCEPTION_ASSERT_LESS(maxwait, 0.02);
    }
}

} // namespace TaskWorker
} // namespace Signal
