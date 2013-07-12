#ifndef SIGNAL_PROCESSING_GETDAGTASKALGORITHM_H
#define SIGNAL_PROCESSING_GETDAGTASKALGORITHM_H

#include "volatileptr.h"
#include "task.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class GetDagTaskAlgorithm: public VolatilePtr<GetDagTaskAlgorithm>
{
public:
    virtual ~GetDagTaskAlgorithm() {}

    virtual Task::Ptr getTask(
            const Graph& g,
            GraphVertex target,
            Signal::Intervals missing_in_target=Intervals::Intervals_ALL,
            Signal::IntervalType center=Interval::IntervalType_MIN,
            Signal::ComputingEngine::Ptr worker=Signal::ComputingEngine::Ptr()) const = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETDAGTASKALGORITHM_H
