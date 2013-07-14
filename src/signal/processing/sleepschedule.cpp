#include "sleepschedule.h"
#include "task.h"

namespace Signal {
namespace Processing {


SleepSchedule::
        SleepSchedule(Bedroom::Ptr bedroom, ISchedule::Ptr schedule)
    :
      bedroom(bedroom),
      schedule(schedule)
{
}


Task::Ptr SleepSchedule::
        getTask() volatile
{
    Task::Ptr task;
    Bedroom::Ptr bedroom;

    {
        ReadPtr that(this);
        const SleepSchedule* self = dynamic_cast<const SleepSchedule*>((const ISchedule*)that);
        bedroom = self->bedroom;
    }

    while (true) {
        {
            ReadPtr that(this);
            const SleepSchedule* self = dynamic_cast<const SleepSchedule*>((const ISchedule*)that);

            if (self->schedule)
                task = self->schedule->getTask();
        }

        if (task)
            return task;

        bedroom->sleep();
    }
}

} // namespace Processing
} // namespace Signal


#include <QThread>

namespace Signal {
namespace Processing {

class ScheduleMock: public ISchedule {
public:
    ScheduleMock() : get_task_calls(0) {}

    Task::Ptr getTask() volatile {
        get_task_calls++;
        if (get_task_calls == 1)
            return Task::Ptr();
        return Task::Ptr(new Task(Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(4,5)));
    }

    int get_task_calls;
};

class WorkerMock: public QThread {
public:
    WorkerMock(ISchedule::Ptr schedule) : schedule(schedule) {}

    virtual void run() {
        schedule->getTask();
    }

    ISchedule::Ptr schedule;
};

void SleepSchedule::
        test()
{
    // It should stall callers while waiting for an available task.
    {
        Bedroom::Ptr bedroom(new Bedroom);
        ISchedule::Ptr schedule_mock(new ScheduleMock);
        ISchedule::Ptr sleep_schedule(new SleepSchedule(bedroom, schedule_mock));

        WorkerMock worker_mock(sleep_schedule);
        worker_mock.start ();

        usleep(1000);
        EXCEPTION_ASSERT(worker_mock.isRunning ());

        bedroom->wakeup();

        usleep(1000);
        EXCEPTION_ASSERT(worker_mock.isFinished ());

        int get_task_calls = dynamic_cast<const ScheduleMock*>(&*read1(schedule_mock))->get_task_calls;
        EXCEPTION_ASSERT_EQUALS(get_task_calls, 2);
    }
}


} // namespace Processing
} // namespace Signal
