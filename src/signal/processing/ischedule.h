#ifndef SIGNAL_PROCESSING_ISCHEDULE_H
#define SIGNAL_PROCESSING_ISCHEDULE_H

#include "signal/computingengine.h"

namespace Signal {
namespace Processing {

class Task;

/**
 * @brief The ISchedule class should provide new tasks for callers who lack
 * additional information.
 *
 * Shall return null if no plausible task was found.
 */
class ISchedule
{
public:
    typedef std::shared_ptr<ISchedule> ptr;

    virtual ~ISchedule() {}

    /**
     * @brief getTask finds if there is something to work on.
     *
     * Note that multiple threads may call getTask simultaneously
     *
     * @return
     */
    virtual Task getTask(Signal::ComputingEngine::ptr engine) const = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_ISCHEDULE_H
