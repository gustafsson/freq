#include "schedulegettask.h"
#include "schedulealgorithm.h"
#include <QThread>

namespace Signal {
namespace Processing {


ScheduleGetTask::
        ScheduleGetTask(Bedroom::Ptr bedroom)
    :
      bedroom(bedroom)
{
}




Task::Ptr ScheduleGetTask::
        getTask() volatile
{
    Task::Ptr task;

    while (true) {
        {
            ReadPtr that(this);
            const ScheduleGetTask* self = dynamic_cast<const ScheduleGetTask*>((const Schedule*)that);

            if (self->get_task)
                task = self->get_task->getTask();
        }

        if (task)
            return task;

        Bedroom::Ptr bedroom;
        {
            ReadPtr that(this);
            const ScheduleGetTask* self = dynamic_cast<const ScheduleGetTask*>((const Schedule*)that);
            bedroom = self->bedroom;
        }

        bedroom->sleep();
    }
}


void ScheduleGetTask::
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
