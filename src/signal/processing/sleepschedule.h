#ifndef SIGNAL_PROCESSING_SCHEDULEGETTASK_H
#define SIGNAL_PROCESSING_SCHEDULEGETTASK_H

#include "dag.h"
#include "task.h"
#include "schedule.h"
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
class SleepSchedule: public Schedule
{
public:
    SleepSchedule(Bedroom::Ptr bedroom, Schedule::Ptr schedule);

    // Sleeps until a task can be returned
    virtual Task::Ptr getTask() volatile;

private:
    Bedroom::Ptr bedroom;
    Schedule::Ptr schedule;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEGETTASK_H
