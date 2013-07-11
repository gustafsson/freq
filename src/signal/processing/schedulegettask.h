#ifndef SIGNAL_PROCESSING_SCHEDULEGETTASK_H
#define SIGNAL_PROCESSING_SCHEDULEGETTASK_H

#include "dag.h"
#include "task.h"
#include "gettask.h"

#include <QMutex>
#include <QWaitCondition>

namespace Signal {
namespace Processing {

class ScheduleGetTask: public GetTask
{
public:
    ScheduleGetTask(Dag::Ptr g);

    virtual Task::Ptr getTask() volatile;
    virtual Task::Ptr getTask();

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
