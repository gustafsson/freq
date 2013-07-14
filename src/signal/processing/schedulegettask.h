#ifndef SIGNAL_PROCESSING_SCHEDULEGETTASK_H
#define SIGNAL_PROCESSING_SCHEDULEGETTASK_H

#include "dag.h"
#include "task.h"
#include "gettask.h"
#include "bedroom.h"

#include <QMutex>
#include <QWaitCondition>

namespace Signal {
namespace Processing {


// Class keep the work condition separate from GetTask.

/**
 * @brief The ScheduleGetTask class should behave as GetTask.
 *
 * It should stall callers while waiting for an available task.
 */
class ScheduleGetTask: public GetTask
{
public:
    ScheduleGetTask(Bedroom::Ptr bedroom);

    GetTask::Ptr getTaskImplementation();
    void updateGetTaskImplementation(GetTask::Ptr);

    // Stalls until a task can be returned
    virtual Task::Ptr getTask() volatile;

private:
    QWaitCondition work_condition;
    QMutex work_condition_mutex;

    GetTask::Ptr get_task;
    Bedroom::Ptr bedroom;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEGETTASK_H
