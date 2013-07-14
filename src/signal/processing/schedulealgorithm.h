#ifndef SIGNAL_PROCESSING_SCHEDULEALGORITHM_H
#define SIGNAL_PROCESSING_SCHEDULEALGORITHM_H

#include "task.h"
#include "dag.h"
#include "workers.h"
#include "signal/computingengine.h"
#include "getdagtaskalgorithm.h"

namespace Signal {
namespace Processing {


/**
 * @brief The ScheduleAlgorithm class should figure out the missing pieces in
 * the graph and produce a Task to work it off
 *
 * It should let missing_in_target override out_of_date in the given vertex
 *
 * Issues
 * Does not know how to cope with workers that doesn't support all steps.
 */
class ScheduleAlgorithm: public GetDagTaskAlgorithm
{
public:
    ScheduleAlgorithm(Workers::Ptr workers = Workers::Ptr());

    Task::Ptr getTask(
            const Graph& g,
            GraphVertex target,
            Signal::Intervals missing_in_target=Intervals::Intervals_ALL,
            Signal::IntervalType center=Interval::IntervalType_MIN,
            Signal::ComputingEngine::Ptr worker=Signal::ComputingEngine::Ptr()) const;

private:
    Workers::Ptr workers_;

public:
    static void test();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEALGORITHM_H
