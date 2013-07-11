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
 * @brief The ScheduleGetTask class should provide new tasks for workers who
 * lack information about what they should do.
 *
 * It should halt works while waiting for an available task.
 */
class ScheduleGetTask: public GetTask
{
public:
    ScheduleGetTask(Dag::Ptr g);

    // Stalls until a task can be returned
    virtual Task::Ptr getTask() volatile;

    // Returns null if no task was found
    virtual Task::Ptr getTask() const;

    // Check if a task might be available
    void wakeup();

private:
    Dag::Ptr g;
    QWaitCondition work_condition;
    QMutex work_condition_mutex;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEGETTASK_H
