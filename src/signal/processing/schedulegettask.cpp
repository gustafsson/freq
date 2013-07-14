#include "schedulegettask.h"
#include "schedulealgorithm.h"
#include <QThread>

namespace Signal {
namespace Processing {


ScheduleGetTask::
        ScheduleGetTask(WorkerBedroom::Ptr worker_bedroom)
    :
      worker_bedroom(worker_bedroom)
{
}




Task::Ptr ScheduleGetTask::
        getTask() volatile
{
    Task::Ptr task;

    while (true) {
        {
            ReadPtr that(this);
            const ScheduleGetTask* self = dynamic_cast<const ScheduleGetTask*>((const GetTask*)that);

            if (self->get_task)
                task = self->get_task->getTask();
        }

        if (task)
            return task;

        WorkerBedroom::Ptr worker_bedroom;
        {
            ReadPtr that(this);
            const ScheduleGetTask* self = dynamic_cast<const ScheduleGetTask*>((const GetTask*)that);
            worker_bedroom = self->worker_bedroom;
        }

        worker_bedroom->sleep();
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
