#include "qteventworkerfactory.h"

#include "signal/processing/task.h"

#include "tasktimer.h"
#include "log.h"
#include "exceptionassert.h"
#include "demangle.h"

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

using namespace Signal::Processing;

namespace Signal {
namespace QtEventWorker {


QtEventWorkerFactory::WorkerWrapper::
        WorkerWrapper(QtEventWorker* p)
    :
      p(p)
{
    EXCEPTION_ASSERT(p);
}


QtEventWorkerFactory::WorkerWrapper::
        ~WorkerWrapper()
{
    // Let Qt delete the QtEventWorker object later in the event loop of the
    // thread that owns the object.
    p->deleteLater ();
}


QtEventWorkerFactory::
        QtEventWorkerFactory(ISchedule::ptr schedule, Bedroom::ptr bedroom)
    :
      schedule_(schedule),
      bedroom_(bedroom),
      notifier_(new BedroomSignalAdapter(bedroom, this))
{
}


QtEventWorkerFactory::
        ~QtEventWorkerFactory()
{
    try {
        notifier_->quit_and_wait ();

        //terminate_workers ();
    } catch (const std::exception& x) {
        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("\nQtEventWorkers::~QtEventWorkers destructor swallowed exception: %s\n"
                                  "%s\n\n")
                    % vartype(x) % boost::diagnostic_information(x)).c_str());
        fflush(stderr);
    } catch (...) {
        fflush(stdout);
        fprintf(stderr, "%s",
                str(boost::format("\nQtEventWorkers::~QtEventWorkers destructor swallowed a non std::exception\n"
                                  "%s\n\n")
                    % boost::current_exception_diagnostic_information ()).c_str());
        fflush(stderr);
    }
}


Signal::Processing::Worker::ptr QtEventWorkerFactory::
        make_worker(Signal::ComputingEngine::ptr ce)
{
    QtEventWorker* w;
    Worker::ptr wp(new WorkerWrapper(w=new QtEventWorker(ce, schedule_, false)));

    bool a = QObject::connect(w,
            SIGNAL(finished(std::exception_ptr,Signal::ComputingEngine::ptr)),
            SIGNAL(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)));
    bool b = connect(notifier_, SIGNAL(wakeup()), w, SLOT(wakeup()));
    bool c = connect(w, SIGNAL(oneTaskDone()), notifier_, SIGNAL(wakeup()));
    bool d = connect(w, SIGNAL(finished(std::exception_ptr,Signal::ComputingEngine::ptr)), notifier_, SIGNAL(wakeup()));

    EXCEPTION_ASSERT(a);
    EXCEPTION_ASSERT(b);
    EXCEPTION_ASSERT(c);
    EXCEPTION_ASSERT(d);

    w->wakeup ();

    return wp;
}


bool QtEventWorkerFactory::
        terminate_workers(Processing::Workers& workers, int timeout)
{
    TIME_TERMINATE TaskTimer ti("terminate_workers");

    workers.remove_all_engines(timeout);

    TIME_TERMINATE TaskInfo("terminate");
    for (auto& i: workers.workers_map()) {
        WorkerWrapper* w = dynamic_cast<WorkerWrapper*>(i.second.get());
        if (w)
            w->terminate ();
    }

    return workers.wait(1000);
}


} // namespace Processing
} // namespace Signal

#include "expectexception.h"
#include "signal/processing/bedroom.h"
#include "timer.h"

#include <QtWidgets> // QApplication
#include <atomic>

namespace Signal {
namespace QtEventWorker {


class BlockScheduleMock: public ISchedule {
protected:
    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        bedroom.wakeup ();

        // This should block the thread and be aborted by QThread::terminate
        dont_return();

        EXCEPTION_ASSERT( false );

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
        for(;;) ::this_thread::sleep_for (std::chrono::microseconds(0)); // Allow OS scheduling to kill the thread (just "for(;;);" would not)
    }
};


void QtEventWorkerFactory::
        test()
{
    std::string name = "QtEventWorkers";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv); // takes 0.4 s if this is the first instantiation of QApplication

    Workers::test ([](ISchedule::ptr schedule){
        Bedroom::ptr bedroom(new Bedroom);
        return IWorkerFactory::ptr(new QtEventWorkerFactory(schedule, bedroom));
    });

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

                Processing::Workers workers(IWorkerFactory::ptr(new QtEventWorkerFactory(s, bedroom)));
                Bedroom::Bed bed = dynamic_cast<BlockScheduleMock*>(s.get ())->bedroom.getBed();

                workers.addComputingEngine(Signal::ComputingEngine::ptr());

                // Wait until the schedule has been called (Bedroom supports
                // that the wakeup in schedule is called even before this sleep call
                // as long as 'bed' is allocated before the wakeup call)
                bed.sleep ();

                EXCEPTION_ASSERT_EQUALS(false, workers.remove_all_engines (10));
                EXCEPTION_ASSERT_EQUALS(true, QtEventWorkerFactory::terminate_workers (workers, 0));

                EXCEPTION_ASSERT_EQUALS(workers.n_workers(), 0u);
                EXPECT_EXCEPTION(QtEventWorker::TerminatedException, workers.rethrow_any_worker_exception ());
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
