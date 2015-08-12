#include "cvworker.h"
#include "log.h"
#include "demangle.h"
#include "signal/processing/task.h"
#include "tasktimer.h"
#include "logtickfrequency.h"

#include <thread>

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

//#define DEBUGINFO
#define DEBUGINFO if(0)

using namespace std;
using namespace Signal::Processing;

namespace Signal {
namespace CvWorker {

CvWorker::CvWorker(
        Signal::ComputingEngine::ptr computing_engine,
        Signal::Processing::Bedroom::ptr bedroom,
        ISchedule::ptr schedule)
    :
      abort_(false),
      bedroom_(bedroom),
      active_time_since_start_(0)
{
    atomic<bool>* abort = &this->abort_;
    std::promise<void> p;
    f = p.get_future ();

    t = std::thread([=,p=std::move(p)]() mutable
    {
#ifdef __GNUC__
        {
            stringstream ss;
            ss << "taskworker" << " " << (computing_engine?vartype(*computing_engine):"(null engine)");
            pthread_setname_np(ss.str ().c_str ());
        }
#endif

        LogTickFrequency ltf_wakeups;
        LogTickFrequency ltf_tasks;
        Timer last_wakeup;

        try {
            Bedroom::Bed b = bedroom->getBed ();

            // run loop at least once to simplify testing when aborting a worker right away
            do {
                Timer work_timer;
                Signal::Processing::Task task;

                {
                    DEBUGINFO TaskTimer tt(boost::format("taskworker: get task %s %s") % vartype(*schedule.get ()) % (computing_engine?vartype(*computing_engine):"(null)") );
                    task = schedule->getTask(computing_engine);
                    active_time_since_start_ += work_timer.elapsedAndRestart ();
                }

                if (task)
                {
                    DEBUGINFO TaskTimer tt(boost::format("taskworker: running task %s") % task.expected_output());
                    task.run();
                    active_time_since_start_ += work_timer.elapsed ();
                    bedroom->wakeup (); // make idle workers wakeup to check if they can do something, won't affect busy workers

                    if (ltf_tasks.tick(false))
                        Log("cvworker: %g wakeups/s, %g tasks/s, activity %.0f%%") % ltf_wakeups.hz () % ltf_tasks.hz () % (100*this->activity ());
                }
                else
                {
                    // when there is nothing to work on, wait until the next frame to check again
                    static const double cap_wakeup_interval = 1./60;
                    double force_sleep_interval = cap_wakeup_interval - last_wakeup.elapsed ();
                    if (force_sleep_interval > 0)
                        std::this_thread::sleep_for (std::chrono::microseconds((long)(force_sleep_interval*1e6)));

                    // wakeup when the dag changes
                    if (!*abort)
                        b.sleep ();
                    last_wakeup.restart ();

                    if (ltf_wakeups.tick(false))
                        Log("cvworker: %g wakeups/s, %g tasks/s, activity %.0f%%") % ltf_wakeups.hz () % ltf_tasks.hz () % (100*this->activity ());
                }
            } while (!*abort);

            p.set_value ();
        } catch (...) {
            p.set_exception (std::current_exception ());
        }
    });
}


CvWorker::~CvWorker() {
    abort();
    join();
}


void CvWorker::abort()
{
    abort_ = true;
    bedroom_->wakeup ();
}


void CvWorker::join()
{
    if (t.joinable ())
    {
        t.join ();
        try {
            f.get ();
        } catch (...) {
            this->caught_exception_ = std::current_exception ();
        }
    }
}


bool CvWorker::wait()
{
    join();
    return true;
}


bool CvWorker::wait(unsigned long ms)
{
    if (!f.valid ())
    {
        // Already joined
        return true;
    }

    if (std::future_status::ready == f.wait_for(std::chrono::milliseconds(ms)))
    {
        join();
        return true;
    }
    return false;
}


bool CvWorker::isRunning()
{
    wait(0);
    return t.joinable ();
}


double CvWorker::activity()
{
    double a = active_time_since_start_ / timer_start_.elapsedAndRestart ();
    active_time_since_start_ = 0;
    return a;
}


std::exception_ptr CvWorker::caught_exception()
{
    return caught_exception_;
}


} // namespace CvWorker
} // namespace Signal

#include "detectgdb.h"
#include "prettifysegfault.h"
#include "expectexception.h"

namespace Signal {
namespace CvWorker {

class GetTaskMock: public ISchedule {
public:
    GetTaskMock() : get_task_count(0) {}

    mutable std::atomic<int> get_task_count;

    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        get_task_count++;
        return Task();
    }
};


class GetTaskSegFaultMock: public ISchedule {
public:
    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        if (DetectGdb::was_started_through_gdb ())
            BOOST_THROW_EXCEPTION(segfault_sigill_exception());

        // Causing deliberate segfault to test that the worker handles it correctly
        // The test verifies that it works to instantiate a TaskInfo works
        TaskInfo("testing instantiated TaskInfo");
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnull-dereference"
        *(int*)0 = 0; // cause segfault
#pragma clang diagnostic pop

        // unreachable code
        return Task();
    }
};


class GetTaskExceptionMock: public ISchedule {
public:
    virtual Task getTask(Signal::ComputingEngine::ptr) const override {
        EXCEPTION_ASSERTX(false, "testing that worker catches exceptions from a scheduler");

        // unreachable code
        return Task();
    }
};


class ImmediateDeadLockMock: public GetTaskMock {
public:
    struct shared_state_traits: shared_state_traits_default {
        double timeout() { return 0.001; }
    };

    virtual Task getTask(Signal::ComputingEngine::ptr engine) const override {
        GetTaskMock::getTask (engine);

        // cause dead lock in 1 ms
        shared_state<ImmediateDeadLockMock> m {new ImmediateDeadLockMock};
        m.write () && m.write ();

        // unreachable code
        return Task();
    }
};


class DeadLockMock: public GetTaskMock {
public:
    virtual Task getTask(Signal::ComputingEngine::ptr engine) const override {
        GetTaskMock::getTask (engine);

        // cause dead lock
        shared_state<DeadLockMock> m {new DeadLockMock};
        m.write () && m.write ();

        // unreachable code
        return Task();
    }
};


class DummySchedule: public ISchedule {
    Task getTask(Signal::ComputingEngine::ptr /*engine*/) const override {
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        return Task(step, std::vector<Step::const_ptr>(), Signal::Operation::ptr(), Signal::Interval(), Signal::Interval() );
    }
};


void CvWorker::
        test()
{
    // It should start and stop automatically
    {
        UNITTEST_STEPS TaskTimer tt("It should start and stop automatically");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        std::this_thread::yield ();
        EXCEPTION_ASSERT( worker.isRunning () );
    }

    // It should run tasks as given by the scheduler
    {
        UNITTEST_STEPS TaskTimer tt("It should run tasks as given by the scheduler");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskMock());
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        worker.wait (1);

        std::this_thread::sleep_for (std::chrono::milliseconds(1000));
        std::this_thread::sleep_for (std::chrono::milliseconds(1000));
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*gettask)->get_task_count );
        // Verify that tasks execute properly in Task::test.

        EXCEPTION_ASSERT( worker.isRunning () );
        worker.abort ();
        EXCEPTION_ASSERT( worker.wait (2) );
        EXCEPTION_ASSERT( !worker.isRunning () );
    }

    // It should wait to be awaken if there are no tasks
    {
        UNITTEST_STEPS TaskTimer tt("It should run tasks as given by the scheduler");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        EXCEPTION_ASSERT( !worker.wait (1) );
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*gettask)->get_task_count );
        this_thread::sleep_for (chrono::milliseconds(1));
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*gettask)->get_task_count );

        bedroom->wakeup ();
        worker.wait (1);
        EXCEPTION_ASSERT_EQUALS( 2, dynamic_cast<GetTaskMock*>(&*gettask)->get_task_count );
    }

    // It should store information about a crashed task (segfault) and stop execution.
    if (false)
    if (!DetectGdb::is_running_through_gdb() && !DetectGdb::was_started_through_gdb ())
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (segfault) and stop execution");

        PrettifySegfault::EnableDirectPrint (false);

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskSegFaultMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        worker.wait (1);
        worker.abort ();
        EXCEPTION_ASSERT( worker.wait (1) );
        EXCEPTION_ASSERT( worker.caught_exception () );

        EXPECT_EXCEPTION(segfault_sigill_exception, rethrow_exception(worker.caught_exception ()));

        PrettifySegfault::EnableDirectPrint (true);
    }

    // It should store information about a crashed task (std::exception) and stop execution. (1)
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (std::exception) and stop execution (1)");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskExceptionMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        QThread::yieldCurrentThread ();
    }

    // It should store information about a crashed task (std::exception) and stop execution. (2)
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (std::exception) and stop execution (2)");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new GetTaskExceptionMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

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

#if !defined SHARED_STATE_NO_TIMEOUT
    // It should store information about a crashed task (LockFailed) and stop execution.
    {
        UNITTEST_STEPS TaskTimer tt("It should store information about a crashed task (LockFailed) and stop execution.");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new ImmediateDeadLockMock);

        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        worker.wait (2);
        worker.abort ();
        EXCEPTION_ASSERT( worker.wait (10) );
        EXCEPTION_ASSERT( worker.caught_exception () );
        EXPECT_EXCEPTION(lock_failed, rethrow_exception(worker.caught_exception ()));

        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*gettask)->get_task_count );
    }
#endif

#ifndef SHARED_STATE_NO_TIMEOUT
    // It should not hang if it causes a deadlock (2)
    {
        UNITTEST_STEPS TaskTimer tt("It should not hang if it causes a deadlock (2)");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new DeadLockMock);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        Log("taskworker %d") % __LINE__;
        EXCEPTION_ASSERT( !worker.wait (1) );
        Log("taskworker %d") % __LINE__;
        worker.abort ();
        Log("taskworker %d") % __LINE__;
        EXCEPTION_ASSERT( worker.wait (200) );
        Log("taskworker %d") % __LINE__;
        worker.wait ();
        Log("taskworker %d") % __LINE__;
        Log("taskworker %d") % __LINE__;
    }
#endif // SHARED_STATE_NO_TIMEOUT

    // It should announce when tasks are finished.
    {
        UNITTEST_STEPS TaskTimer tt("It should announce when tasks are finished.");

        Bedroom::ptr bedroom(new Bedroom);
        ISchedule::ptr gettask(new DummySchedule);
        CvWorker worker(Signal::ComputingEngine::ptr(), bedroom, gettask);

        bedroom->wakeup ();
        Bedroom::Bed b = bedroom->getBed ();

        bool aborted_from_timeout = !b.sleep ( 100 );
        EXCEPTION_ASSERT(!aborted_from_timeout);
   }

    // It should wake up sleeping workers when any work is done to see if they can
    // help out on what's left.
    {

    }
}

} // namespace CvWorker
} // namespace Signal
