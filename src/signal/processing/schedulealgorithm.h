#ifndef SIGNAL_PROCESSING_SCHEDULEALGORITHM_H
#define SIGNAL_PROCESSING_SCHEDULEALGORITHM_H

#include "volatileptr.h"
#include "task.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class ScheduleAlgorithm: public VolatilePtr<ScheduleAlgorithm>
{
public:
    virtual ~ScheduleAlgorithm() {}

    virtual Task::Ptr getTask(
            const Graph& g,
            GraphVertex target,
            Signal::Intervals missing_in_target=Intervals::Intervals_ALL,
            Signal::IntervalType center=Interval::IntervalType_MIN,
            Signal::ComputingEngine::Ptr worker=Signal::ComputingEngine::Ptr()) const = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEALGORITHM_H
