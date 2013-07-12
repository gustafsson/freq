#ifndef SIGNAL_PROCESSING_SCHEDULEGETTASK_H
#define SIGNAL_PROCESSING_SCHEDULEGETTASK_H

#include "dag.h"
#include "task.h"
#include "gettask.h"

#include <QMutex>
#include <QWaitCondition>

namespace Signal {
namespace Processing {



/**
 * @brief The ScheduleGetTask class should behave as GetTask.
 *
 * It should stall callers while waiting for an available task.
 */
class ScheduleGetTask: public GetTask
{
public:
    ScheduleGetTask();

    GetTask::Ptr getTaskImplementation();
    void updateGetTaskImplementation(GetTask::Ptr);

    // Check if a task might be available
    void wakeup();

    // Stalls until a task can be returned
    virtual Task::Ptr getTask() volatile;

private:
    QWaitCondition work_condition;
    QMutex work_condition_mutex;

    GetTask::Ptr get_task;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEGETTASK_H
