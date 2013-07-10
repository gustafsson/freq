#ifndef SIGNAL_PROCESSING_SCHEDULE_H
#define SIGNAL_PROCESSING_SCHEDULE_H

#include "task.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class Schedule
{
public:
    Schedule();

    Task::Ptr getTask(
            Graph& g,
            GraphVertex target,
            Signal::Intervals missing_in_target=Intervals::Intervals_ALL,
            int preferred_size=1<<16,
            Signal::IntervalType center=Interval::IntervalType_MIN);

    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULE_H
