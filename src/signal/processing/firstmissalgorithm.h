#ifndef SIGNAL_PROCESSING_FIRSTMISSALGORITHM_H
#define SIGNAL_PROCESSING_FIRSTMISSALGORITHM_H

#include "task.h"
#include "dag.h"
#include "workers.h"
#include "signal/computingengine.h"
#include "ischedulealgorithm.h"

namespace Signal {
namespace Processing {


/**
 * @brief The ScheduleAlgorithm class should figure out the missing pieces in
 * the graph and produce a Task to work it off
 *
 * It should let missing_in_target override out_of_date in the given vertex
 *
 * Issues
 * ------
 * Has not thoroughly tested how it copes with workers that doesn't support all
 * steps. If the engine does not support the selected step the task will
 * silently ignore it.
 *
 * The idea is that if an OperationDesc doesn't support a ComputingEngine this
 * thread will then have to wait, or do something else. Some other thread will
 * have to populate the cache instead.
 *
 * If an OperationDesc doesn't support any of the current ComputingEngines no
 * work will get done.
 *
 * Todo
 * ----
 * To utilize workers that doesn't support all steps FirstMissAlgorithm could
 * keep on breath_first_searching until a supported step is found.
 */
class FirstMissAlgorithm: public IScheduleAlgorithm
{
public:
    Task::ptr getTask(
            const Graph& g,
            GraphVertex target,
            Signal::Intervals needed=Intervals::Intervals_ALL,
            Signal::IntervalType center=Interval::IntervalType_MIN,
            Signal::IntervalType preferred_size=Interval::IntervalType_MAX,
            Workers::ptr workers=Workers::ptr(),
            Signal::ComputingEngine::ptr worker=Signal::ComputingEngine::ptr()) const;

public:
    static void test();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_FIRSTMISSALGORITHM_H
