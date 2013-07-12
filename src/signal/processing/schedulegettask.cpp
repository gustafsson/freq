#include "schedulegettask.h"
#include "schedulealgorithm.h"
#include <QThread>

namespace Signal {
namespace Processing {


ScheduleGetTask::
        ScheduleGetTask()
{
}


GetTask::Ptr ScheduleGetTask::
        getTaskImplementation()
{
    return get_task;
}


void ScheduleGetTask::
        updateGetTaskImplementation(GetTask::Ptr value)
{
    get_task = value;
    wakeup();
}


void ScheduleGetTask::
        wakeup()
{
    work_condition.wakeAll ();
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

        // QWaitCondition/QMutex are thread-safe so we can discard the volatile qualifier
        const_cast<QWaitCondition*>(&work_condition)->wait (
                    const_cast<QMutex*> (&work_condition_mutex));
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
