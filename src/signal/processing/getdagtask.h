#ifndef SIGNAL_PROCESSING_GETDAGTASK_H
#define SIGNAL_PROCESSING_GETDAGTASK_H

#include "getdagtaskalgorithm.h"
#include "gettask.h"

namespace Signal {
namespace Processing {

/**
 * @brief The ScheduleGetTask class should behave as GetTask.
 *
 * It should return return null if no plausible task was found.
 */
class GetDagTask: public GetTask {
public:
    GetDagTask(Dag::Ptr g, GetDagTaskAlgorithm::Ptr algorithm);

    virtual Task::Ptr getTask() volatile;

private:
    Dag::Ptr g;
    GetDagTaskAlgorithm::Ptr algorithm;

    Target::Ptr prioritizedTarget() const;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETDAGTASK_H
