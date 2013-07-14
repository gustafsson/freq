#include "sleepschedule.h"
#include "ischedulealgorithm.h"
#include <QThread>

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


void SleepSchedule::
        test()
{
    // It should provide new tasks for workers who lack information about what they should do
    {

    }

    // It should halt works while waiting for an available task
    {

    }
}


} // namespace Processing
} // namespace Signal
