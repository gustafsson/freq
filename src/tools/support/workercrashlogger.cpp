#include "workercrashlogger.h"

#include "TaskTimer.h"
#include "demangle.h"
#include "exceptionassert.h"

#include <QTimer>

//#define DEBUG
#define DEBUG if(0)

using namespace Signal::Processing;

namespace Tools {
namespace Support {

class DummyException: virtual public boost::exception, virtual public std::exception {};

WorkerCrashLogger::
        WorkerCrashLogger(Workers::Ptr workers, bool consume_exceptions)
    :
      workers_(workers),
      consume_exceptions_(consume_exceptions)
{
    moveToThread (&thread_);
    thread_.start ();

    Workers::WritePtr ww(workers);
    // Log any future worker crashes
    connect(&*ww,
            SIGNAL(worker_quit(boost::exception_ptr,Signal::ComputingEngine::Ptr)),
            SLOT(worker_quit(boost::exception_ptr,Signal::ComputingEngine::Ptr)));

    // Log previous worker crashes. As the new thread owns this, 'check' will
    // be executed in the new thread.
    if (consume_exceptions)
        QTimer::singleShot (0, this, SLOT(check_all_previously_crashed_and_consume()));
    else
        QTimer::singleShot (0, this, SLOT(check_all_previously_crashed_without_consuming()));
}


WorkerCrashLogger::
        ~WorkerCrashLogger()
{
    thread_.quit ();
    thread_.wait ();
}


void WorkerCrashLogger::
        worker_quit(boost::exception_ptr e, Signal::ComputingEngine::Ptr ce)
{
    DEBUG TaskInfo ti("WorkerCrashLogger::worker_quit");
    if (consume_exceptions_)
    {
        check_all_previously_crashed_and_consume();
    }
    else
    {
        try {
            rethrow_exception(e);
        } catch ( const DummyException& x) {
            DEBUG TaskInfo ti("got x");
            boost::diagnostic_information(x);
        } catch ( const boost::exception& x) {
            x << Workers::crashed_engine_value(ce);
            TaskInfo(boost::format("Worker '%s' crashed\n%s") % (ce?vartype(*ce):"(null)") % boost::diagnostic_information(x));
        }
    }
}


void WorkerCrashLogger::
        check_all_previously_crashed_and_consume ()
{
    DEBUG TaskInfo ti("WorkerCrashLogger::check_previously_crashed_and_consume");

    try {
        write1(workers_)->rethrow_any_worker_exception();

        return;
    } catch ( const DummyException& x) {
        DEBUG TaskInfo ti("got x");
        // Force the slow backtrace beautifier
        boost::diagnostic_information(x);
    } catch ( const boost::exception& x) {
        Signal::ComputingEngine::Ptr ce;
        if( Signal::ComputingEngine::Ptr const * mi=boost::get_error_info<Workers::crashed_engine_value>(x) )
            ce = *mi;

        TaskInfo(boost::format("Worker '%s' crashed\n%s") % (ce?vartype(*ce):"(null)") % boost::diagnostic_information(x));
    }

    check_all_previously_crashed_and_consume();
}


void WorkerCrashLogger::
        check_all_previously_crashed_without_consuming()
{
    DEBUG TaskInfo ti("check_all_previously_crashed_without_consuming");
    Workers::EngineWorkerMap workers_map = read1(workers_)->workers_map();
    for(Workers::EngineWorkerMap::const_iterator i=workers_map.begin (); i != workers_map.end(); ++i) {
        QPointer<Worker> worker = i->second;
        if (worker && !worker->isRunning ())
            worker_quit (worker->caught_exception (), i->first);
    }
}

} // namespace Support
} // namespace Tools

#include <QApplication>
#include "timer.h"

namespace Signal { namespace Processing { class Task; }}

namespace Tools {
namespace Support {

class DummyScheduler: public ISchedule
{
    boost::shared_ptr<volatile Task> getTask(Signal::ComputingEngine::Ptr) volatile
    {
        // Throw exception
        BOOST_THROW_EXCEPTION(DummyException());
    }
};


void addAndWaitForStop(Workers::Ptr workers)
{
    QEventLoop e;
    QObject::connect(&*write1(workers),
            SIGNAL(worker_quit(boost::exception_ptr,Signal::ComputingEngine::Ptr)),
            &e, SLOT(quit()));
    write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
    e.exec ();
}


void WorkerCrashLogger::
        test()
{
    // It should fetch information asynchronously of crashed workers.
    {
        DEBUG TaskInfo ti("Catch info from a previously crashed worker");

        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv);

        Timer timer;

        //for (int consume=0; consume<2; consume++)
        ISchedule::Ptr schedule(new DummyScheduler);
        Workers::Ptr workers(new Workers(schedule));

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);
        addAndWaitForStop(workers);


        {
            WorkerCrashLogger wcl(workers);
            a.processEvents (); // Init new thread before telling it to quit

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, 5e-3 );

            // When the thread quits. Wait for the beautifier to log everything.
        }

        // Should have consumed all workers
        Workers::DeadEngines de = write1(workers)->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0);

        // Should have taken a while (the backtrace beautifier is slow)
        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 1e-5, T );
        EXCEPTION_ASSERT_LESS( T, 1e-3 );
    }

    {
        DEBUG TaskInfo ti("Catch info from a crashed worker as it happens");

        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv);

        Timer timer;

        ISchedule::Ptr schedule(new DummyScheduler);
        Workers::Ptr workers(new Workers(schedule));

        {
            WorkerCrashLogger wcl(workers);

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);
            addAndWaitForStop(workers);

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, 2e-3 );
        }

        Workers::DeadEngines de = write1(workers)->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0);

        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 1e-5, T );
        EXCEPTION_ASSERT_LESS( T, 1e-4 );
    }

    {
        DEBUG TaskInfo ti("Support not consuming workers");

        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv);

        Timer timer;

        ISchedule::Ptr schedule(new DummyScheduler);
        Workers::Ptr workers(new Workers(schedule));

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);

        {
            WorkerCrashLogger wcl(workers, false);

            a.processEvents ();

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, 2e-3 );
        }

        // Should not have consumed any workers
        Workers::DeadEngines de = write1(workers)->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 2);

        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 0.1e-5, T );
        EXCEPTION_ASSERT_LESS( T, 2e-4 );
    }
}


} // namespace Support
} // namespace Tools
