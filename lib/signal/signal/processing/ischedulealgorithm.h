#ifndef SIGNAL_PROCESSING_ISCHEDULEALGORITHM_H
#define SIGNAL_PROCESSING_ISCHEDULEALGORITHM_H

#include <memory>
#include "task.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class IScheduleAlgorithm
{
public:
    typedef std::unique_ptr<IScheduleAlgorithm> ptr;

    virtual ~IScheduleAlgorithm() {}

    virtual Task getTask(
            const Graph& g,
            GraphVertex target,
            Signal::Intervals needed, //=Intervals::Intervals_ALL,
            Signal::IntervalType center, //=Interval::IntervalType_MIN,
            Signal::IntervalType preferred_size, //=Interval::IntervalType_MAX,
            Signal::ComputingEngine::ptr worker) const = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_ISCHEDULEALGORITHM_H
