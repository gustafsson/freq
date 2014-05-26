// Include QObject and Boost.Foreach in that order to prevent conflicts with Qt foreach
#include <QObject>
#include <boost/foreach.hpp>

#include "workers.h"
#include "targetschedule.h"
#include "timer.h"
#include "bedroomsignaladapter.h"
#include "demangle.h"
#include "tasktimer.h"

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

namespace Signal {
namespace Processing {


Workers::
        Workers(ISchedule::ptr schedule, Bedroom::ptr bedroom)
    :
      schedule_(schedule),
      notifier_(new BedroomSignalAdapter(bedroom, this))
{
}


Workers::
        ~Workers()
{
    try {
        schedule_.reset ();
        notifier_->quit_and_wait ();

        //terminate_workers ();
        remove_all_engines ();

        print (clean_dead_workers());
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


Worker::ptr Workers::
        addComputingEngine(Signal::ComputingEngine::ptr ce)
{
    if (workers_map_.find (ce) != workers_map_.end ())
        EXCEPTION_ASSERTX(false, "Engine already added");

    Worker::ptr w(new Worker(ce, schedule_));
    workers_map_[ce] = w;

    updateWorkers();
    bool a = connect(&*w,
            SIGNAL(finished(std::exception_ptr,Signal::ComputingEngine::ptr)),
            SIGNAL(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)));
    bool b = connect(notifier_, SIGNAL(wakeup()), &*w, SLOT(wakeup()));
    bool c = connect(&*w, SIGNAL(oneTaskDone()), notifier_, SIGNAL(wakeup()));
    bool d = connect((Worker*)&*w, SIGNAL(finished(std::exception_ptr,Signal::ComputingEngine::ptr)), notifier_, SIGNAL(wakeup()));

    EXCEPTION_ASSERT(a);
    EXCEPTION_ASSERT(b);
    EXCEPTION_ASSERT(c);
    EXCEPTION_ASSERT(d);

    return w;
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
          }
      }

    updateWorkers();
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

    for(EngineWorkerMap::const_iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        Worker::ptr worker = i->second;

        if (worker && worker->isRunning ())
            N++;
    }

    return N;
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

    for (EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        Worker::ptr worker = i->second;

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
        rethrow_any_worker_exception()
{
    for (EngineWorkerMap::iterator i=workers_map_.begin (); i != workers_map_.end(); ++i) {
        Worker::ptr worker = i->second;

        if (worker) {
            std::exception_ptr e = worker->caught_exception ();
            if (e) {
                Signal::ComputingEngine::ptr ce = i->first;

                worker->deleteLater ();
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
        wait(int timeout)
{
    TIME_TERMINATE TaskTimer ti("wait(%d)", timeout);

    bool ok = true;

    BOOST_FOREACH(EngineWorkerMap::value_type i, workers_map_) {
        if (i.second)
            ok &= i.second->wait (timeout);
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
        if (i.second) i.second->abort();
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

#include "expectexception.h"
#include "bedroom.h"

#include <QApplication>
#include <atomic>

namespace Signal {
namespace Processing {

class GetEmptyTaskMock: public ISchedule {
public:
    GetEmptyTaskMock() : get_task_count(0) {}

    mutable std::atomic<int> get_task_count;

    virtual Task::ptr getTask(Signal::ComputingEngine::ptr) const override {
        //__sync_fetch_and_add (&self->get_task_count, 1);

        // could also use 'boost::detail::atomic_count get_task_count;'
        int v = get_task_count.fetch_add(1);
        if (v%2)
            throw std::logic_error("test crash");
        else
            return Task::ptr();
    }
};


class BlockScheduleMock: public ISchedule {
protected:
    virtual Task::ptr getTask(Signal::ComputingEngine::ptr) const override {
        bedroom.wakeup ();

        // This should block the thread and be aborted by QThread::terminate
        dont_return();

        EXCEPTION_ASSERT( false );

        return Task::ptr();
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


void Workers::
        test()
{
    std::string name = "Workers";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv); // takes 0.4 s if this is the first instantiation of QApplication

    {
        UNITTEST_STEPS TaskInfo("It should start and stop computing engines as they are added and removed");

        // It should start and stop computing engines as they are added and removed
        double maxwait = 0;
        for (int j=0;j<100; j++){
            ISchedule::ptr schedule(new GetEmptyTaskMock);
            Bedroom::ptr bedroom(new Bedroom);
            Workers workers(schedule, bedroom);
            workers.rethrow_any_worker_exception(); // Should do nothing

            Timer t;
            int worker_count = 40; // Number of threads to start. Multiplying by 10 multiplies the elapsed time by a factor of 100.
            std::list<Worker::ptr> workerlist;
            Worker::ptr w = workers.addComputingEngine(Signal::ComputingEngine::ptr());
            workerlist.push_back (w);
            for (int i=1; i<worker_count; ++i) {
                Worker::ptr w = workers.addComputingEngine(Signal::ComputingEngine::ptr(new Signal::ComputingCpu));
                workerlist.push_back (w);
            }

            // Wait until they're done
            BOOST_FOREACH (Worker::ptr& w, workerlist) { w->abort (); w->wait (); }
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

            Workers::DeadEngines dead = workers.clean_dead_workers ();
            Engines engines = workers.workers();

            EXCEPTION_ASSERT_EQUALS(engines.size (), 0u);
            EXCEPTION_ASSERT_EQUALS(dead.size (), (size_t)worker_count-1); // One was cleared by catching its exception above

            // When dead workers are cleared there should not be any exceptions thrown
            workers.rethrow_any_worker_exception();
        }

//        EXCEPTION_ASSERT_LESS(maxwait, 0.02);
    }

    {
        UNITTEST_STEPS TaskInfo("It should terminate all threads when it's closed");

        ISchedule::ptr schedule[] = {
            ISchedule::ptr(new SleepScheduleMock),
            ISchedule::ptr(new LockScheduleMock),
            ISchedule::ptr(new BusyScheduleMock)
        };

        for (unsigned i=0; i<sizeof(schedule)/sizeof(schedule[0]); i++) {
            Timer t;
            {
                ISchedule::ptr s = schedule[i];
                //TaskInfo ti(boost::format("%s") % vartype(*s));
                Bedroom::ptr bedroom(new Bedroom);

                Workers workers(s, bedroom);
                Bedroom::Bed bed = dynamic_cast<BlockScheduleMock*>(s.get ())->bedroom.getBed();

                workers.addComputingEngine(Signal::ComputingEngine::ptr());

                // Wait until the schedule has been called (Bedroom supports
                // that the wakeup in schedule is called even before this sleep call
                // as long as 'bed' is allocated before the wakeup call)
                bed.sleep ();

                EXCEPTION_ASSERT_EQUALS(false, workers.remove_all_engines (10));
                EXCEPTION_ASSERT_EQUALS(true, workers.terminate_workers (0));

                EXCEPTION_ASSERT_EQUALS(workers.n_workers(), 0u);
                EXPECT_EXCEPTION(Worker::TerminatedException, workers.rethrow_any_worker_exception ());
                workers.clean_dead_workers ();
            }
            float elapsed = t.elapsed ();
            float n = (i+1)*0.00001;
            EXCEPTION_ASSERT_LESS(0.01+n, elapsed);
            EXCEPTION_ASSERT_LESS(elapsed, 0.04+n); // +n makes it possible to see in the test log which iteration that failed
        }
    }

    // It should wake up sleeping workers when any work is done to see if they can
    // help out on what's left.
    {

    }
}


} // namespace Processing
} // namespace Signal
