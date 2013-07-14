#ifndef SIGNAL_PROCESSING_WORKERREADY_H
#define SIGNAL_PROCESSING_WORKERREADY_H

#include <QMutex>
#include <QWaitCondition>

#include "volatileptr.h"

namespace Signal {
namespace Processing {

/**
 * @brief The WorkerBedroom class
 *
 * Issues
 * WorkerBedroom
 */
class WorkerBedroom: public VolatilePtr<WorkerBedroom>
{
public:
    // Check if a task might be available
    void wakeup();

    void sleep() volatile;

private:
    QWaitCondition work_condition;
    QMutex work_condition_mutex;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERREADY_H
