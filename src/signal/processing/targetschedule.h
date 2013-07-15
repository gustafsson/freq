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
    TargetSchedule(Dag::Ptr g, IScheduleAlgorithm::Ptr algorithm, Targets::Ptr targets);

    virtual Task::Ptr getTask() volatile;

private:
    Targets::Ptr targets;

    Dag::Ptr g;
    IScheduleAlgorithm::Ptr algorithm;

    TargetNeeds::Ptr prioritizedTarget() const;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETSCHEDULE_H
