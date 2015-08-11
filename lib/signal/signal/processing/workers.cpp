#include "workers.h"
#include "signal/processing/targetschedule.h"
#include "timer.h"
#include "signal/processing/bedroomsignaladapter.h"
#include "demangle.h"
#include "tasktimer.h"

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

namespace Signal {
namespace Processing {


Workers::Workers(IWorkerFactory::ptr&& workerfactory)
    :
      workerfactory_(move(workerfactory))
{

}


Workers::
        ~Workers()
{
    try {
        remove_all_engines ();

        print (clean_dead_workers());

        workers_map_.clear ();

        workerfactory_.reset ();

        workers_.clear ();
    } catch (const std::exception& x) {
        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("\nWorkers::~Workers destructor swallowed exception: %s\n"
                                  "%s\n\n")
                    % vartype(x) % boost::diagnostic_information(x)).c_str());
        fflush(stderr);
    } catch (...) {
        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("\nWorkers::~Workers destructor swallowed a non std::exception\n"
                                  "%s\n\n")
                    % boost::current_exception_diagnostic_information ()).c_str());
        fflush(stderr);
    }
}


void Workers::
        addComputingEngine(Signal::ComputingEngine::ptr ce)
{
    if (workers_map_.find (ce) != workers_map_.end ())
        EXCEPTION_ASSERTX(false, "Engine already added");

    Worker::ptr wp = workerfactory_->make_worker(ce);
    workers_map_.insert (make_pair(ce,move(wp)));

    updateWorkers();
}


void Workers::
        removeComputingEngine(Signal::ComputingEngine::ptr ce)
{
    EngineWorkerMap::iterator worker = workers_map_.find (ce);
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


const Workers::Engines& Workers::
        workers() const
{
    return workers_;
}


const Workers::EngineWorkerMap& Workers::
        workers_map() const
{
    return workers_map_;
}


size_t Workers::
        n_workers() const
{
    size_t N = 0;

    for(const auto& i : workers_map_) {
        const Worker::ptr& worker = i.second;

        if (worker && worker->isRunning ())
            N++;
    }

    return N;
}


bool Workers::
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


bool Workers::
        remove_all_engines(int timeout) const
{
    TIME_TERMINATE TaskTimer ti("remove_all_engines");

    // Make sure the workers doesn't start anything new
    for(const auto& i: workers_map_) {
        if (i.second) i.second->abort();
    }

    return wait(timeout);
}


void Workers::
        updateWorkers()
{
    Engines engines;

    for (const auto& ewp: workers_map_)
        engines.push_back (ewp.first);

    workers_ = move(engines);
}


Workers::DeadEngines Workers::
        clean_dead_workers()
{
    DeadEngines dead;

    for (EngineWorkerMap::iterator i=workers_map_.begin ();
         i != workers_map_.end();
         ++i)
    {
        const Worker::ptr& worker = i->second;

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


void Workers::
        rethrow_any_worker_exception()
{
    for (EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        Worker::ptr& worker = i->second;

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


void Workers::
        print(const DeadEngines& engines)
{
    if (engines.empty ())
        return;

    TaskInfo ti("Dead engines");

    for (const DeadEngines::value_type& e : engines) {
        Signal::ComputingEngine::ptr engine = e.first;
        std::exception_ptr x = e.second;
        std::string enginename = engine ? vartype(*engine.get ()) : (vartype(engine.get ())+"==0");

        if (x)
        {
            std::string details;
            try {
                std::rethrow_exception(x);
            } catch(...) {
                details = boost::current_exception_diagnostic_information();
            }

            TaskInfo(boost::format("engine %1% failed.\n%2%")
                     % enginename
                     % details);
        }
        else
        {
            TaskInfo(boost::format("engine %1% stopped")
                     % enginename);
        }
    }
}


} // namespace Processing
} // namespace Signal

namespace Signal {
namespace Processing {

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


void Workers::
    test(std::function<IWorkerFactory::ptr(ISchedule::ptr)> workerfactoryfactory)
{
    {
        UNITTEST_STEPS TaskInfo("It should start and stop computing engines as they are added and removed");

        // It should start and stop computing engines as they are added and removed
        double maxwait = 0;
        for (int j=0;j<100; j++){
            ISchedule::ptr schedule(new GetEmptyTaskMock);
            Processing::Workers workers(workerfactoryfactory(schedule));
            workers.rethrow_any_worker_exception(); // Should do nothing

            Timer t;
            int worker_count = 40; // Number of threads to start. Multiplying by 10 multiplies the elapsed time by a factor of 100.
            workers.addComputingEngine(Signal::ComputingEngine::ptr());
            for (int i=1; i<worker_count; ++i)
                workers.addComputingEngine(Signal::ComputingEngine::ptr(new Signal::ComputingCpu));

            std::list<Worker*> workerlist;
            for(const auto& v : workers.workers_map())
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
                        boost::get_error_info<Workers::crashed_engine>(x);
                EXCEPTION_ASSERT(ce);

                const std::string* cename =
                        boost::get_error_info<Workers::crashed_engine_typename>(x);
                EXCEPTION_ASSERT(cename);
            }

            Workers::DeadEngines dead = workers.clean_dead_workers ();
            Workers::Engines engines = workers.workers();

            EXCEPTION_ASSERT_EQUALS(engines.size (), 0u);
            EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)worker_count-1); // One was cleared by catching its exception above

            // When dead workers are cleared there should not be any exceptions thrown
            workers.rethrow_any_worker_exception();
        }

//        EXCEPTION_ASSERT_LESS(maxwait, 0.02);
    }
}

} // namespace Processing
} // namespace Signal
