#ifndef TASKWORKERS_H
#define TASKWORKERS_H

#include "signal/processing/worker.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/ischedule.h"

namespace Signal {
namespace CvWorker {

/**
 * @brief The CvWorkers class should consume 200+ tasks per second without
 * running a thread sleep-awake loop between each task. Causing threads to
 * sleep that often is inefficient use of resources and not allowed on iOS
 * (iOS will kill an app that averages 150+ wakeups per second over 10
 * minutes).
 */
class CvWorkerFactory: public Processing::IWorkerFactory
{
public:
    CvWorkerFactory(Processing::ISchedule::ptr schedule, Processing::Bedroom::ptr bedroom);

    Processing::Worker::ptr make_worker(Signal::ComputingEngine::ptr ce) override;

private:
    Processing::ISchedule::ptr schedule_;
    Processing::Bedroom::ptr bedroom_;

public:
    static void test();
};

} // namespace CvWorker
} // namespace Signal

#endif // TASKWORKERS_H
