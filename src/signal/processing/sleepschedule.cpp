#include "sleepschedule.h"
#include "task.h"

#include <QThread>

//#define DEBUGINFO
#define DEBUGINFO if(0)

namespace Signal {
namespace Processing {


SleepSchedule::
        SleepSchedule(Bedroom::Ptr bedroom, ISchedule::Ptr schedule)
    :
      bedroom_(bedroom),
      schedule_(schedule)
{
}


Task::Ptr SleepSchedule::
        getTask() volatile
{
    Bedroom::Ptr bedroom;
    ISchedule::Ptr schedule;

    {
        ReadPtr that(this);
        const SleepSchedule* self = (const SleepSchedule*)&*that;

        bedroom = self->bedroom_;
        schedule = self->schedule_;
    }

    // Wake up TargetNeeds::sleep... random hack that depends on the behaviour of "everything"...
    // It would make more sense if TargetNeeds monitored when things got updated in the target.
    bedroom->wakeup ();

    Bedroom::Bed bed = bedroom->getBed();
    for (;;) try {
        {
            DEBUGINFO TaskTimer tt(boost::format("Searching for a task"));

            Task::Ptr task = schedule->getTask();

            if (task)
                return task;

            DEBUGINFO TaskInfo(boost::format("Didn't find a task. Going to bed"));
        }
        bed.sleep();
        DEBUGINFO TaskInfo(boost::format("Woke up"));
    } catch (const Signal::Processing::BedroomClosed&) {
        return Task::Ptr();
    } catch (const std::exception&) {
        // Ask another worker to try instead
        bedroom->wakeup ();
        throw;
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
        if (get_task_calls <= 1)
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
        EXCEPTION_ASSERT_EQUALS(get_task_calls, 2);
    }
}


} // namespace Processing
} // namespace Signal
