#include "workercrashlogger.h"

#include "tasktimer.h"
#include "demangle.h"
#include "exceptionassert.h"
#include "signal/processing/task.h"
#include "tools/applicationerrorlogcontroller.h"

#include <QTimer>

//#define DEBUG
#define DEBUG if(0)

using namespace Signal::Processing;

namespace Tools {
namespace Support {

class DummyException: virtual public boost::exception, virtual public std::exception {};


WorkerCrashLogger::
        WorkerCrashLogger(Workers::ptr workers, bool consume_exceptions)
    :
      workers_(workers),
      consume_exceptions_(consume_exceptions)
{
    moveToThread (&thread_);
    // Remove responsibility for event processing for this when the the thread finishes
    connect(&thread_, SIGNAL(finished()), SLOT(finished()));
    thread_.start ();

    auto ww = workers.write ();
    // Log any future worker crashes
    connect(&*ww,
            SIGNAL(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)),
            SLOT(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)));

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
    TaskInfo ti("~WorkerCrashLogger");
    thread_.quit ();
    thread_.wait ();
}


void WorkerCrashLogger::
        worker_quit(std::exception_ptr e, Signal::ComputingEngine::ptr ce)
{
    DEBUG TaskInfo ti(boost::format("WorkerCrashLogger::worker_quit %s") % (e?"normally":"with exception"));
    if (consume_exceptions_)
      {
        workers_.write ()->removeComputingEngine(ce);
      }

    try
      {
        if (e)
          {
            rethrow_exception(e);
          }

      }
    catch ( const boost::exception& x)
      {
        x << Workers::crashed_engine(ce) << Workers::crashed_engine_typename(ce?vartype(*ce):"(null)");

        log(x);
      }
}


void WorkerCrashLogger::
        check_all_previously_crashed_and_consume ()
{
    DEBUG TaskInfo ti("WorkerCrashLogger::check_previously_crashed_and_consume");

    while(true) try
      {
        workers_.write ()->rethrow_any_worker_exception();

        break;
      }
    catch ( const boost::exception& x)
      {
        log(x);
      }
}


void WorkerCrashLogger::
        check_all_previously_crashed_without_consuming()
{
    DEBUG TaskInfo ti("check_all_previously_crashed_without_consuming");

    Workers::EngineWorkerMap workers_map = workers_.read ()->workers_map();

    for(Workers::EngineWorkerMap::const_iterator i=workers_map.begin (); i != workers_map.end(); ++i)
      {
        Worker::ptr worker = i->second;

        if (worker && !worker->isRunning ())
          {
            worker_quit (worker->caught_exception (), i->first);
          }
      }
}


void WorkerCrashLogger::
        finished()
{
    moveToThread (0);
}


void WorkerCrashLogger::
        log(const boost::exception& x)
{
    // Fetch various info from the exception to make a prettier log
    std::string crashed_engine_typename = "<no info>";
    std::string operation_desc_text;

    if( std::string const * mi = boost::get_error_info<Workers::crashed_engine_typename>(x) )
      {
        crashed_engine_typename = *mi;
      }

    if( Step::ptr const * mi = boost::get_error_info<Step::crashed_step>(x) )
      {
        Signal::OperationDesc::ptr od;
        {
            auto s = mi->write ();
            s->mark_as_crashed_and_get_invalidator ();
            od = s->get_crashed ();
        }

        if (od)
        {
            auto o = od.read ();
            Signal::Processing::IInvalidator::ptr i = o->getInvalidator();
            i.read ()->deprecateCache (Signal::Intervals::Intervals_ALL);
            operation_desc_text = " in " + o->toString().toStdString();
        }
      }

    if( Signal::Interval const * mi = boost::get_error_info<Task::crashed_expected_output>(x) )
      {
        operation_desc_text += " " + mi->toString ();
      }


    // Ignore logging in UnitTest
    if (dynamic_cast<const DummyException*>(&x))
      {
        // Execute to_string for all tagged info (i.e force the slow backtrace beautifier if it's included)
        boost::diagnostic_information(x);
        return;
      }

    TaskInfo(boost::format("1 of %d workers crashed: '%s'%s")
             % workers_.read ()->n_workers()
             % crashed_engine_typename
             % operation_desc_text);

    bool send_feedback = true;
    if (send_feedback)
        ApplicationErrorLogController::registerException (boost::current_exception ());
    else
        TaskInfo(boost::format("%s") % boost::diagnostic_information(x));
}


} // namespace Support
} // namespace Tools

#include <QApplication>
#include "timer.h"
#include "detectgdb.h"

namespace Signal { namespace Processing { class Task; }}

namespace Tools {
namespace Support {

class DummyScheduler: public ISchedule
{
    Task::ptr getTask(Signal::ComputingEngine::ptr) const override
    {
        // Throw exception
        BOOST_THROW_EXCEPTION(DummyException());
    }
};


void addAndWaitForStop(Workers::ptr workers)
{
    QEventLoop e;
    QObject::connect(&*workers.write (),
            SIGNAL(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)),
            &e, SLOT(quit()));
    workers.write ()->addComputingEngine(Signal::ComputingEngine::ptr(new Signal::ComputingCpu));
    e.exec ();
}


void WorkerCrashLogger::
        test()
{
    bool gdb = DetectGdb::is_running_through_gdb();

    // It should fetch information asynchronously of crashed workers.
    {
        DEBUG TaskInfo ti("Catch info from a previously crashed worker");

        std::string name = "WorkerCrashLogger1";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        Timer timer;

        //for (int consume=0; consume<2; consume++)
        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new Workers(schedule, bedroom));

        {
            WorkerCrashLogger wcl(workers);
        }

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);
        addAndWaitForStop(workers);


        {
            WorkerCrashLogger wcl(workers);
            a.processEvents (); // Init new thread before telling it to quit

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, 15e-3 );

            // When the thread quits. Wait for the beautifier to log everything.
        }

        // Should have consumed all workers
        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0u);

        // Should have taken a while (the backtrace beautifier is slow)
        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 1e-5, T );
        EXCEPTION_ASSERT_LESS( T, 1e-3 );
    }

    {
        DEBUG TaskInfo ti("Catch info from a crashed worker as it happens");

        std::string name = "WorkerCrashLogger2";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        Timer timer;

        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new Workers(schedule, bedroom));

        {
            WorkerCrashLogger wcl(workers);

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);
            addAndWaitForStop(workers);

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, 3e-3 );
        }

        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0u);

        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 1e-5, T );
        EXCEPTION_ASSERT_LESS( T, 2e-4 );
    }

    {
        DEBUG TaskInfo ti("Support not consuming workers");

        std::string name = "WorkerCrashLogger3";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        Timer timer;

        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new Workers(schedule, bedroom));

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);

        {
            WorkerCrashLogger wcl(workers, false);

            a.processEvents ();

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);

            double T = timer.elapsedAndRestart ();
            EXCEPTION_ASSERT_LESS( T, gdb ? 20e-3 : 4e-3 );
        }

        // Should not have consumed any workers
        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 2u);

        double T = timer.elapsedAndRestart ();
        EXCEPTION_ASSERT_LESS( 1e-6, T );
        EXCEPTION_ASSERT_LESS( T, 300e-6 );
    }
}


} // namespace Support
} // namespace Tools
