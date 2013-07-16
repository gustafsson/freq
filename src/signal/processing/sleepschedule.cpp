#include "sleepschedule.h"
#include "task.h"

namespace Signal {
namespace Processing {


SleepSchedule::
        SleepSchedule(Bedroom::Ptr bedroom, ISchedule::Ptr schedule)
    :
      bedroom(bedroom),
      schedule(schedule),
      enough(false)
{
}


SleepSchedule::
        ~SleepSchedule()
{
    enough = true;
    bedroom->wakeup();
    bedroom->sleep();
}


Task::Ptr SleepSchedule::
        getTask() volatile
{
    Bedroom::Ptr bedroom;
    ISchedule::Ptr schedule;

    {
        ReadPtr that(this);
        const SleepSchedule* self = (const SleepSchedule*)&*that;

        bedroom = self->bedroom;
        schedule = self->schedule;
    }

    for (;;) {
        Task::Ptr task = schedule->getTask();

        if (task || enough) {
            bedroom->wakeup();
            return task;
        }

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

        EXCEPTION_ASSERT_EQUALS(worker_mock.wait (1), false);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers(), 1);

        bedroom->wakeup();

        EXCEPTION_ASSERT_EQUALS(worker_mock.wait (1), true);

        int get_task_calls = dynamic_cast<const ScheduleMock*>(&*read1(schedule_mock))->get_task_calls;
        EXCEPTION_ASSERT_EQUALS(get_task_calls, 2);
    }
}


} // namespace Processing
} // namespace Signal
