#include "bedroom.h"

namespace Signal {
namespace Processing {

void Bedroom::
        wakeup()
{
    work_condition.wakeAll ();
}


void Bedroom::
        sleep() volatile
{
    // QWaitCondition/QMutex are thread-safe so we can discard the volatile qualifier
    const_cast<QWaitCondition*>(&work_condition)->wait (
                const_cast<QMutex*> (&work_condition_mutex));
}


} // namespace Processing
} // namespace Signal
