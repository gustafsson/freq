#ifndef SIGNAL_PROCESSING_ISCHEDULE_H
#define SIGNAL_PROCESSING_ISCHEDULE_H

#include "shared_state.h"
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
    typedef shared_state<ISchedule> Ptr;

    virtual ~ISchedule() {}

    /**
     * @brief getTask finds if there is something to work on.
     *
     * Note that multiple threads may call getTask simultaneously
     *
     * @return
     */
    virtual shared_state<Task> getTask(Signal::ComputingEngine::Ptr engine) const = 0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_ISCHEDULE_H
