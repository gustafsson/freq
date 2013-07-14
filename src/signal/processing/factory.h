#ifndef SIGNAL_PROCESSING_FACTORY_H
#define SIGNAL_PROCESSING_FACTORY_H

#include "gettask.h"
#include "dag.h"
#include "invalidator.h"

namespace Signal {
namespace Processing {

class Factory
{
public:
    Factory();

    // Dag::Ptr dag;
    GetTask::Ptr getTargetTasks; // getdagtask
    GetTask::Ptr waitForItGetTasks; // ScheduleGetTask
    Invalidator::Ptr invalidator;

    getTarget(Step::Ptr)
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_FACTORY_H
