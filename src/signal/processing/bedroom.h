#ifndef SIGNAL_PROCESSING_WORKERREADY_H
#define SIGNAL_PROCESSING_WORKERREADY_H

#include <QMutex>
#include <QWaitCondition>

#include "volatileptr.h"

namespace Signal {
namespace Processing {

/**
 * @brief The WorkerBedroom class should allow different threads to sleep on
 * this object until another thread calls wakeup().
 */
class Bedroom: public VolatilePtr<Bedroom>
{
public:
    Bedroom();

    // Check if a task might be available
    void wakeup() volatile;

    void sleep() volatile;

    int sleepers() const volatile;
private:
    int sleepers_;
    QWaitCondition work_condition;
    QMutex work_condition_mutex;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERREADY_H
