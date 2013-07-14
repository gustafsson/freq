#include "worker.h"

namespace Signal {
namespace Processing {

Worker::
        Worker (Signal::ComputingEngine::Ptr computing_eninge, Schedule::Ptr get_task)
    :
      computing_eninge_(computing_eninge),
      get_task_(get_task),
      enough_(false),
      exception_type_(0)
{
}


void Worker::
        run()
{
    try {
        Task::Ptr task;

        while (task = get_task_->getTask())
        {
            task->run(computing_eninge_);

            if (enough_)
                break;
        }
    } catch (const std::exception& x) {
        exception_what_ = x.what();
        const std::type_info& t = typeid(x);
        exception_type_ = &t;
    }

    deleteLater ();
}


void Worker::
        exit_nicely_and_delete()
{
    enough_ = true;
}


const std::string& Worker::
        exception_what() const
{
    return exception_what_;
}


const std::type_info* Worker::
        exception_type() const
{
    return exception_type_;
}


class GetTaskMock: public Schedule {
public:
    GetTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        get_task_count++;
        return Task::Ptr();
    }
};


class GetTaskSegFaultMock: public Schedule {
public:
    virtual Task::Ptr getTask() volatile {
        int a = *(int*)0; // cause segfault
        a=a;
        return Task::Ptr();
    }
};


class GetTaskExceptionMock: public Schedule {
public:
    virtual Task::Ptr getTask() volatile {
        EXCEPTION_ASSERTX(false, "GetTaskExceptionMock");
        return Task::Ptr();
    }
};


void Worker::
        test()
{
    // It should run the next task as long as there is one
    {
        Schedule::Ptr gettask(new GetTaskMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.start ();
        bool finished = worker.wait (3); // Stop running within 3 ms, equals !worker.isRunning ()
        EXCEPTION_ASSERT_EQUALS( true, finished );

        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<GetTaskMock*>(&*write1(gettask))->get_task_count );
        // Verify that tasks execute properly in Task::test.
    }

    // It should store information about a crashed task (segfault)
    {
        Schedule::Ptr gettask(new GetTaskSegFaultMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.run ();
        bool finished = worker.wait (2);
        EXCEPTION_ASSERT_EQUALS( true, finished );

        const std::type_info* ti = worker.exception_type();

        EXCEPTION_ASSERT_EQUALS( demangle (ti?ti->name ():""), demangle (typeid(SignalException).name ()) );
    }

    // It should store information about a crashed task (C++ exception)
    {
        Schedule::Ptr gettask(new GetTaskExceptionMock());

        Worker worker(Signal::ComputingEngine::Ptr(), gettask);
        worker.run ();
        bool finished = worker.wait (2);
        EXCEPTION_ASSERT_EQUALS( true, finished );

        const std::type_info* ti = worker.exception_type();

        EXCEPTION_ASSERT_EQUALS( demangle (ti?ti->name ():""), demangle (typeid(ExceptionAssert).name ()) );
        EXCEPTION_ASSERT_EQUALS( "GetTaskExceptionMock", worker.exception_what () );
    }
}


} // namespace Processing
} // namespace Signal
