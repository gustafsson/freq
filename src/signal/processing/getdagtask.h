#ifndef SIGNAL_PROCESSING_GETDAGTASK_H
#define SIGNAL_PROCESSING_GETDAGTASK_H

#include "getdagtaskalgorithm.h"
#include "gettask.h"
#include "targetinvalidator.h"
#include "targets.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GetDagTask class should provide tasks to keep a Dag up-to-date with respect to all targets.
 */
class GetDagTask: public GetTask {
public:
    GetDagTask(Dag::Ptr g, GetDagTaskAlgorithm::Ptr algorithm, Targets::Ptr targets);

    virtual Task::Ptr getTask() volatile;

private:
    Targets::Ptr targets;

    Dag::Ptr g;
    GetDagTaskAlgorithm::Ptr algorithm;

    Target::Ptr prioritizedTarget() const;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETDAGTASK_H
