#ifndef SIGNAL_PROCESSING_GETDAGTASK_H
#define SIGNAL_PROCESSING_GETDAGTASK_H

#include "getdagtaskalgorithm.h"
#include "gettask.h"
#include "targetinvalidator.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GetDagTask class should provide tasks to keep a Dag up-to-date with respect to all targets.
 */
class GetDagTask: public GetTask {
public:
    GetDagTask(Dag::Ptr g, GetDagTaskAlgorithm::Ptr algorithm, std::vector<Target::Ptr> targets);

    TargetUpdater::Ptr addTarget(Step::Ptr step);
    void removeTarget(Step::Ptr step);
    std::vector<Step::Ptr> getTargets() const;

    // Targets have a timestamp, the target with the latest timestamp is computed first.
    // Targets are free to compete with different prioritizes by setting arbitrary timestamps.
    // This list is publicly accesible.



    virtual Task::Ptr getTask() volatile;

private:

    typedef std::vector<Target::Ptr> TargetInfos;
    TargetInfos targets;

    Dag::Ptr g;
    GetDagTaskAlgorithm::Ptr algorithm;

    Target::Ptr prioritizedTarget() const;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETDAGTASK_H
