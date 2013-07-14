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

    virtual boost::shared_ptr<volatile Task> getTask() volatile=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_ISCHEDULE_H
