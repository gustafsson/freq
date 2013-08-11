#include "worker.h"
#include "task.h"
#include "detectgdb.h"

namespace Signal {
namespace Processing {

Worker::
        Worker (Signal::ComputingEngine::Ptr computing_eninge, ISchedule::WeakPtr schedule)
    :
      computing_eninge_(computing_eninge),
      schedule_(schedule)
{
}


void Worker::
        run()
{
    try {

        for (;;)
        {
            ISchedule::Ptr schedule = schedule_.lock ();
            if (!schedule)
                break;

            Task::Ptr task = schedule->getTask();
            if (!task)
                break;

            task->run(computing_eninge_);
        }
    } catch (const std::exception&) {
        exception_ = boost::current_exception ();
    }

    deleteLater ();
}


void Worker::
        exit_nicely_and_delete()
{
    schedule_.reset ();
}


boost::exception_ptr Worker::
        caught_exception() const
{
    return exception_;
}


class GetTaskMock: public ISchedule {
public:
    GetTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        get_task_count++;
        return Task::Ptr();
    }
};


class GetTaskSegFaultMock: public ISchedule {
public:
    virtual Task::Ptr getTask() volatile {
        if (DetectGdb::was_started_through_gdb ())
            BOOST_THROW_EXCEPTION(SignalException(SIGSEGV));

        TaskInfo("Causing deliberate segfault to test that the worker handles it correctly");
        int a = *(int*)0; // cause segfault
        a=a;
        return Task::Ptr();
    }
};


class GetTaskExceptionMock: public ISchedule {
public:
    virtual Task::Ptr getTask() volatile {
        EXCEPTION_ASSERTX(false, "testing that worker catches exceptions from a scheduler");
        return Task::Ptr();
    }
};


void Worker::
        test()
{
    // It should run the next task as long as there is one
    {
        ISchedule::Ptr gettask(new GetTaskMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.start ();
        bool finished = worker.wait (3); // Stop running within 3 ms, equals !worker.isRunning ()
        EXCEPTION_ASSERT_EQUALS( true, finished );

        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );
        // Verify that tasks execute properly in Task::test.
    }

    // It should store information about a crashed task (segfault)
    {
        ISchedule::Ptr gettask(new GetTaskSegFaultMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.run ();
        bool finished = worker.wait (2);
        EXCEPTION_ASSERT_EQUALS( true, finished );
        EXCEPTION_ASSERT( worker.caught_exception () );

        try {
            rethrow_exception(worker.caught_exception ());
            BOOST_THROW_EXCEPTION(boost::unknown_exception());
        } catch (const SignalException&) {
        }
    }

    // It should store information about a crashed task (C++ exception)
    {
        ISchedule::Ptr gettask(new GetTaskExceptionMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.run ();
        bool finished = worker.wait (2);
        EXCEPTION_ASSERT_EQUALS( true, finished );
        EXCEPTION_ASSERT( worker.caught_exception () );

        try {
            rethrow_exception(worker.caught_exception ());
            BOOST_THROW_EXCEPTION(boost::unknown_exception());
        } catch (const ExceptionAssert& x) {
            const std::string* message = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x);
            EXCEPTION_ASSERT_EQUALS( "testing that worker catches exceptions from a scheduler", message?*message:"" );
        }
    }
}


} // namespace Processing
} // namespace Signal
