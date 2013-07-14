#ifndef SIGNAL_PROCESSING_GETTASK_H
#define SIGNAL_PROCESSING_GETTASK_H

#include "task.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GetTask class should provide new tasks for workers who lack
 * information about what they should do.
 *
 * It may return null if no plausible task was found.
 * It may block the calling thread until a plausible task is found.
 */
class Schedule: public VolatilePtr<Schedule>
{
public:
    virtual ~Schedule() {}

    virtual Task::Ptr getTask() volatile=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETTASK_H
