#include "workercrashlogger.h"

#include "tasktimer.h"
#include "demangle.h"
#include "exceptionassert.h"
#include "signal/processing/task.h"
#include "tools/applicationerrorlogcontroller.h"
#include "signal/pollworker/pollworkers.h"

#include <QTimer>

//#define DEBUG
#define DEBUG if(0)

using namespace Signal::Processing;
using namespace Signal::PollWorker;

namespace Tools {
namespace Support {

class DummyException: virtual public boost::exception, virtual public std::exception {};


WorkerCrashLogger::
        WorkerCrashLogger(Workers::ptr workers, bool consume_exceptions)
    :
      workers_(workers),
      consume_exceptions_(consume_exceptions)
{
    EXCEPTION_ASSERTX(dynamic_cast<Signal::PollWorker::PollWorkers*>(workers.raw ()), "WorkerCrashLogger only supports PollWorkers");

    moveToThread (&thread_);
    // Remove responsibility for event processing for this when the the thread finishes
    connect(&thread_, SIGNAL(finished()), SLOT(finished()));
    thread_.start ();

    auto ww = workers.write ();
    // Log any future worker crashes
    connect(dynamic_cast<Signal::PollWorker::PollWorkers*>(ww.get()),
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

    PollWorkers::EngineWorkerMap workers_map = dynamic_cast<const PollWorkers*>(workers_.read ().get ())->workers_map();

    for(PollWorkers::EngineWorkerMap::const_iterator i=workers_map.begin (); i != workers_map.end(); ++i)
      {
        PollWorker::ptr worker = i->second;

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
            o->getInvalidator()->deprecateCache (Signal::Intervals::Intervals_ALL);
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

#include <QtWidgets> // QApplication
#include "timer.h"
#include "trace_perf.h"

namespace Signal { namespace Processing { class Task; }}

namespace Tools {
namespace Support {

class DummyScheduler: public ISchedule
{
    Task getTask(Signal::ComputingEngine::ptr) const override
    {
        // Throw exception
        BOOST_THROW_EXCEPTION(DummyException());
    }
};


void addAndWaitForStop(Workers::ptr workers)
{
    QEventLoop e;
    // Log any future worker crashes
    QObject::connect(dynamic_cast<Signal::PollWorker::PollWorkers*>(workers.write ().get()),
            SIGNAL(worker_quit(std::exception_ptr,Signal::ComputingEngine::ptr)),
            &e, SLOT(quit()));

    workers.write ()->addComputingEngine(Signal::ComputingEngine::ptr(new Signal::ComputingCpu));
    e.exec ();
}


void WorkerCrashLogger::
        test()
{
    // It should fetch information asynchronously of crashed workers.
    {
        DEBUG TaskInfo ti("Catch info from a previously crashed worker");

        std::string name = "WorkerCrashLogger1";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        //for (int consume=0; consume<2; consume++)
        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new PollWorkers(schedule, bedroom));

        {
            WorkerCrashLogger wcl(workers);
        }

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);
        addAndWaitForStop(workers);

        {
            TRACE_PERF("Init");

            WorkerCrashLogger wcl(workers);
            a.processEvents (); // Init new thread before telling it to quit

            // When the thread quits. Wait for the beautifier to log everything.
        }

        // Should have consumed all workers
        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0u);
    }

    {
        DEBUG TaskInfo ti("Catch info from a crashed worker as it happens");

        std::string name = "WorkerCrashLogger2";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new PollWorkers(schedule, bedroom));

        {
            TRACE_PERF("Catch info from a crashed worker as it happens");

            WorkerCrashLogger wcl(workers);

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);
            addAndWaitForStop(workers);
        }

        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 0u);
    }

    {
        DEBUG TaskInfo ti("Support not consuming workers");

        std::string name = "WorkerCrashLogger3";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        ISchedule::ptr schedule(new DummyScheduler);
        Bedroom::ptr bedroom(new Bedroom);
        Workers::ptr workers(new PollWorkers(schedule, bedroom));

        // Catch info from a previously crashed worker
        addAndWaitForStop(workers);

        {
            TRACE_PERF("Support not consuming workers");

            WorkerCrashLogger wcl(workers, false);

            a.processEvents ();

            // Catch info from a crashed worker as it happens
            addAndWaitForStop(workers);
        }

        // Should not have consumed any workers
        Workers::DeadEngines de = workers.write ()->clean_dead_workers();
        EXCEPTION_ASSERT_EQUALS(de.size (), 2u);
    }
}


} // namespace Support
} // namespace Tools
