#include "sleepschedule.h"
#include "task.h"

#include <QThread>

//#define DEBUGINFO
#define DEBUGINFO if(0)

namespace Signal {
namespace Processing {


SleepSchedule::
        SleepSchedule(Bedroom::WeakPtr bedroom, ISchedule::Ptr schedule)
    :
      bedroom_(bedroom),
      schedule_(schedule)
{
}


Task::Ptr SleepSchedule::
        getTask() volatile
{
    Bedroom::WeakPtr wbedroom;
    ISchedule::Ptr schedule;

    {
        ReadPtr that(this);
        const SleepSchedule* self = (const SleepSchedule*)&*that;

        wbedroom = self->bedroom_;
        schedule = self->schedule_;

        Bedroom::Ptr bedroom = wbedroom.lock ();
        if (bedroom) {
            DEBUGINFO TaskInfo("wakeup");
            bedroom->wakeup();
        }
    }

    for (;;) {
        Bedroom::Ptr bedroom = wbedroom.lock ();
        if (!bedroom)
            return Task::Ptr();

        DEBUGINFO TaskInfo(boost::format("starts searching for a task"));

        Task::Ptr task = schedule->getTask();

        if (task) {
            bedroom->wakeup();
            return task;
        }

        DEBUGINFO TaskInfo(boost::format("didn't find a task. Going to bed"));
        bedroom->sleep();
        DEBUGINFO TaskInfo(boost::format("woke up"));
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
        if (get_task_calls <= 2)
            return Task::Ptr();
        return Task::Ptr(new Task(0, Step::Ptr(), std::vector<Step::Ptr>(), Signal::Interval(4,5)));
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

        EXCEPTION_ASSERT(!worker_mock.wait (1));
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers(), 1);

        bedroom->wakeup();

        EXCEPTION_ASSERT(worker_mock.wait (1));

        int get_task_calls = dynamic_cast<const ScheduleMock*>(&*read1(schedule_mock))->get_task_calls;
        EXCEPTION_ASSERT_EQUALS(get_task_calls, 3);
    }
}


} // namespace Processing
} // namespace Signal
