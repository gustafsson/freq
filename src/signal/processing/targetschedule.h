#ifndef SIGNAL_PROCESSING_TARGETSCHEDULE_H
#define SIGNAL_PROCESSING_TARGETSCHEDULE_H

#include "ischedulealgorithm.h"
#include "ischedule.h"
#include "targets.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GetDagTask class should provide tasks to keep a Dag up-to-date with respect to all targets.
 */
class TargetSchedule: public ISchedule {
public:
    // Requires workers and/or current worker
    TargetSchedule(Dag::ptr g, IScheduleAlgorithm::ptr algorithm, Targets::ptr targets);

    virtual Task::ptr getTask(Signal::ComputingEngine::ptr engine) const;

private:
    Targets::ptr targets;

    Dag::ptr g;
    IScheduleAlgorithm::ptr algorithm;

    TargetNeeds::ptr::read_ptr prioritizedTarget() const;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETSCHEDULE_H
