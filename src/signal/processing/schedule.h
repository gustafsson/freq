#ifndef SIGNAL_PROCESSING_SCHEDULE_H
#define SIGNAL_PROCESSING_SCHEDULE_H

#include "task.h"
#include "dag.h"
#include "worker.h"

namespace Signal {
namespace Processing {

class Schedule
{
public:
    Schedule(Dag::Ptr g);

    Task::Ptr getTask();

    QWaitCondition work_condition;

    std::list<Worker> workers;
    // List workers that belongs to this scheduler

private:
    Dag::Ptr g;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULE_H
