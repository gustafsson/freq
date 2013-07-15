#ifndef SIGNAL_PROCESSING_ISCHEDULE_H
#define SIGNAL_PROCESSING_ISCHEDULE_H

#include "volatileptr.h"

namespace Signal {
namespace Processing {

class Task;

/**
 * @brief The ISchedule class should provide new tasks for callers who lack
 * additional information.
 *
 * It may return null if no plausible task was found.
 * It may block the calling thread until a plausible task is found.
 */
class ISchedule: public VolatilePtr<ISchedule>
{
public:
    virtual ~ISchedule() {}

    /**
     * @brief getTask finds if there is something to work on.
     *
     * Note that multiple threads may call getTask simultaneously (see VolatilePtr)
     *
     * @return
     */
    virtual boost::shared_ptr<volatile Task> getTask() volatile=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_ISCHEDULE_H
