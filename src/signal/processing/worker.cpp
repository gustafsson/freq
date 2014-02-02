#include "worker.h"
#include "task.h"
#include "TaskTimer.h"
#include "demangle.h"

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

//#define DEBUGINFO
#define DEBUGINFO if(0)

namespace Signal {
namespace Processing {

bool enable_lockfailed_print = true;

class QTerminatableThread : public QThread {
public:
    void run() {
        setTerminationEnabled ();
        QThread::run ();
    }
};


Worker::
        Worker (Signal::ComputingEngine::Ptr computing_engine, ISchedule::Ptr schedule)
    :
      computing_engine_(computing_engine),
      schedule_(schedule),
      thread_(new QTerminatableThread),
      exception_(new AtomicValue<boost::exception_ptr>)
{
    EXCEPTION_ASSERTX(QThread::currentThread ()->eventDispatcher (),
                      "Worker uses a QThread with an event loop. The QEventLoop requires QApplication");

    qRegisterMetaType<boost::exception_ptr>("boost::exception_ptr");
    qRegisterMetaType<Signal::ComputingEngine::Ptr>("Signal::ComputingEngine::Ptr");

    // Create terminated_exception_
    try {
        // To make caught_exception() non-zero if the thread is terminated even
        // though no exact information about the crash reason is stored. The
        // log file might contain more details.
        BOOST_THROW_EXCEPTION(Worker::TerminatedException());
    } catch (...) {
        terminated_exception_ = boost::current_exception ();
    }

    // Start the worker thread as an event based background thread
    thread_->setParent (this);
    thread_->start (QThread::LowPriority);
    moveToThread (thread_);

    connect (thread_, SIGNAL(finished()), SLOT(finished()));

    // Initial check to see if work can begin right away
    wakeup (); // will be dispatched to execute in thread_
}


Worker::
        ~Worker ()
{
    abort ();
    wait (1); // To quit the thread normally if idle (returns within 1 ms if it is ready to quit)
    terminate ();
    wait (100);
}


void Worker::
        abort()
{
    thread_->requestInterruption ();
    wakeup ();
}


void Worker::
        terminate()
{
    thread_->terminate ();
}


bool Worker::
        isRunning() const
{
    return thread_->isRunning ();
}


bool Worker::
        wait(unsigned long time)
{
    return thread_->wait (time);
}


boost::exception_ptr Worker::
        caught_exception() const
{
    if (isRunning ())
        return boost::exception_ptr();
    return *exception_;
}


void Worker::
        wakeup()
  {
    if (QThread::currentThread () != this->thread ())
      {
        // Dispatch
        QMetaObject::invokeMethod (this, "wakeup");
        return;
      }

    DEBUGINFO TaskInfo("Worker::wakeup");
    try
      {
        // Let exception_ mark unexpected termination.
        *exception_ = terminated_exception_;

        loop_while_tasks();

        // Finished normal execution without any exception.
        *exception_ = boost::exception_ptr();
      }
    catch (...)
      {
        *exception_ = boost::current_exception ();
        QThread::currentThread ()->requestInterruption ();
      }

    if (QThread::currentThread ()->isInterruptionRequested ())
      {
        QThread::currentThread ()->quit ();
      }
  }


void Worker::
        finished()
  {
    DEBUGINFO TaskInfo("Worker::finished");
    moveToThread (0); // important. otherwise 'thread_' will try to delete 'this', but 'this' owns 'thread_' -> crash.
    emit finished(*exception_, computing_engine_);
  }


void Worker::
        loop_while_tasks()
  {
    Task::Ptr task;
    int consecutive_lock_failed_count = 0;
    while (!QThread::currentThread ()->isInterruptionRequested ())
      {
        try
          {
            {
                DEBUGINFO TaskTimer tt(boost::format("Get task %s %s") % vartype(*schedule_) % (computing_engine_?vartype(*computing_engine_):"(null)") );
                task = schedule_->getTask(computing_engine_);
            }

            if (task)
              {
                DEBUGINFO TaskTimer tt(boost::format("Running task %s") % read1(task)->expected_output());
                write1(task)->run(computing_engine_);
                emit oneTaskDone();
              }
            else
              {
                // Wait for a new wakeup call
                break;
              }

            consecutive_lock_failed_count = 0;
          }
        catch (const LockFailed& x)
          {
            if (enable_lockfailed_print)
              {
                TaskInfo("");
                TaskInfo("Lock failed");
                TaskInfo(boost::format("%s") % boost::diagnostic_information(x));
                TaskInfo("");
              }

            if (consecutive_lock_failed_count < 1)
              {
                consecutive_lock_failed_count++;
                if (enable_lockfailed_print)
                    TaskInfo("Starting attempt %d", consecutive_lock_failed_count+1);
              }
            else
                throw;
          }
      }
  }


} // namespace Processing
} // namespace Signal

#include <QApplication>
#include "detectgdb.h"
#include "prettifysegfault.h"
#include "expectexception.h"

#include <QTimer>

namespace Signal {
namespace Processing {

class GetTaskMock: public ISchedule {
public:
    GetTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask(Signal::ComputingEngine::Ptr) volatile {
        get_task_count++;
        return Task::Ptr();
    }
};


class GetTaskSegFaultMock: public ISchedule {
public:
    virtual Task::Ptr getTask(Signal::ComputingEngine::Ptr) volatile {
        if (DetectGdb::was_started_through_gdb ())
            BOOST_THROW_EXCEPTION(segfault_exception());

        // Causing deliberate segfault to test that the worker handles it correctly
        // The test verifies that it works to instantiate a TaskInfo works
        TaskInfo("testing instantiated TaskInfo");
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-dereference"
        *(int*)0 = 0; // cause segfault
#pragma clang diagnostic pop

        // unreachable code
        return Task::Ptr();
    }
};


class GetTaskExceptionMock: public ISchedule {
public:
    virtual Task::Ptr getTask(Signal::ComputingEngine::Ptr) volatile {
        EXCEPTION_ASSERTX(false, "testing that worker catches exceptions from a scheduler");

        // unreachable code
        return Task::Ptr();
    }
};


class DeadLockMock: public GetTaskMock {
public:
    virtual Task::Ptr getTask(Signal::ComputingEngine::Ptr engine) volatile {
        GetTaskMock::getTask (engine);

        // cause dead lock in 1 ms
        volatile DeadLockMock m;
        WritePtr(&m, 1).get() && WritePtr(&m, 1).get();

        // unreachable code
        return Task::Ptr();
    }
};


class DummyTask: public Task {
public:
    DummyTask() : Task(0, Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(), Signal::Interval()) {}

    void run(Signal::ComputingEngine::Ptr) override {
        // Keeps on running a lot of tasks as fast as possible
    }
};

class DummySchedule: public ISchedule {
    Task::Ptr getTask(Signal::ComputingEngine::Ptr engine) volatile override {
        return Task::Ptr(new DummyTask);
    }
};

void Worker::
        test()
{
    std::string name = "Worker";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);

    // It should start and stop automatically
    {
        UNITTEST_STEPS TaskTimer tt("It should start and stop automatically");

        ISchedule::Ptr gettask(new GetTaskMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        QThread::yieldCurrentThread ();
        EXCEPTION_ASSERT( worker.isRunning () );
    }

    // It should run tasks as given by the scheduler
    {
        UNITTEST_STEPS TaskTimer tt("It should run tasks as given by the scheduler");

        ISchedule::Ptr gettask(new GetTaskMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        worker.wait (1);

        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );
        // Verify that tasks execute properly in Task::test.

        EXCEPTION_ASSERT( worker.isRunning () );
        worker.abort ();
        worker.wait (1);
        EXCEPTION_ASSERT( !worker.isRunning () );
    }

    // It should wait to be awaken if there are no tasks
    {
        UNITTEST_STEPS TaskTimer tt("It should run tasks as given by the scheduler");

        ISchedule::Ptr gettask(new GetTaskMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        EXCEPTION_ASSERT( !worker.wait (1) );
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );
        QThread::msleep (1);
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );

        worker.wakeup ();
        worker.wait (1);
        EXCEPTION_ASSERT_EQUALS( 2, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );
    }

    // It should store information about a crashed task (segfault) and stop execution.
#ifdef _DEBUG
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (segfault) and stop execution");

        PrettifySegfault::EnableDirectPrint (false);

        ISchedule::Ptr gettask(new GetTaskSegFaultMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        worker.wait (1);
        worker.abort ();
        EXCEPTION_ASSERT( worker.wait (1) );
        EXCEPTION_ASSERT( worker.caught_exception () );

        EXPECT_EXCEPTION(segfault_exception, rethrow_exception(worker.caught_exception ()));

        PrettifySegfault::EnableDirectPrint (true);
    }
#endif

    // It should store information about a crashed task (std::exception) and stop execution. (1)
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (std::exception) and stop execution (1)");

        ISchedule::Ptr gettask(new GetTaskExceptionMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        QThread::yieldCurrentThread ();
    }

    // It should store information about a crashed task (std::exception) and stop execution. (2)
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (std::exception) and stop execution (2)");

        ISchedule::Ptr gettask(new GetTaskExceptionMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        worker.wait (1);
        worker.abort ();
        worker.wait (1);
        worker.abort ();

        EXCEPTION_ASSERT( worker.caught_exception () );

        try {
            rethrow_exception(worker.caught_exception ());
            BOOST_THROW_EXCEPTION(boost::unknown_exception());
        } catch (const ExceptionAssert& x) {
            const std::string* message = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x);
            EXCEPTION_ASSERT_EQUALS( "testing that worker catches exceptions from a scheduler", message?*message:"" );
        }
    }

    // It should not hang if it causes a deadlock (1)
    {
        UNITTEST_STEPS TaskTimer tt("It should not hang if it causes a deadlock (1)");

        enable_lockfailed_print = false;

        ISchedule::Ptr gettask(new DeadLockMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        worker.wait (10);
        worker.terminate ();
        worker.terminate ();
        worker.terminate ();
        worker.terminate ();
        worker.abort ();
        worker.abort ();
        worker.abort ();
        worker.abort ();

        enable_lockfailed_print = true;
    }

    // It should not hang if it causes a deadlock (2)
    {
        UNITTEST_STEPS TaskTimer tt("It should not hang if it causes a deadlock (2)");

        enable_lockfailed_print = false;

        ISchedule::Ptr gettask(new DeadLockMock());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        EXCEPTION_ASSERT( !worker.wait (1) );
        worker.terminate ();
        EXCEPTION_ASSERT( worker.wait (2) );
        EXPECT_EXCEPTION( TerminatedException, rethrow_exception(worker.caught_exception ()) );

        enable_lockfailed_print = true;
    }


    // It should swallow one LockFailed without aborting the thread, but abort
    // if several consecutive LockFailed are thrown.
    {
        UNITTEST_STEPS TaskTimer tt("It should swallow one LockFailed without aborting the thread, but abort if several consecutive LockFailed are thrown.");

        enable_lockfailed_print = false;

        ISchedule::Ptr gettask(new DeadLockMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);

        worker.wait (2);
        worker.abort ();
        EXCEPTION_ASSERT( worker.wait (10) );
        EXCEPTION_ASSERT( worker.caught_exception () );

        EXPECT_EXCEPTION(LockFailed, rethrow_exception(worker.caught_exception ()));

        EXCEPTION_ASSERT_EQUALS( 2, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );

        enable_lockfailed_print = true;
    }

    // It should announce when tasks are finished.
    {
        UNITTEST_STEPS TaskTimer tt("It should announce when tasks are finished.");

        ISchedule::Ptr gettask(new DummySchedule());
        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        QEventLoop e;
        QTimer t;
        connect (&t, SIGNAL(timeout()), &e, SLOT(quit()));
        connect (&worker, SIGNAL(oneTaskDone()), &e, SLOT(quit()));
        t.setSingleShot( true );
        t.setInterval( 0 );
        t.start();
        e.exec ();
        EXCEPTION_ASSERT(t.isActive());
   }
}


} // namespace Processing
} // namespace Signal
