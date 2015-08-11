#ifndef TASKWORKERS_H
#define TASKWORKERS_H

#include <memory>
#include "signal/processing/workers.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/ischedule.h"

namespace Signal {
namespace TaskWorker {

/**
 * @brief The TaskWorkers class should consume 200+ tasks per second without
 * running a thread sleep-awake loop between each task. Causing threads to
 * sleep that often is inefficient use of resources and not allowed on iOS
 * (iOS will kill an app that averages 150+ wakeups per second over 10
 * minutes).
 */
class TaskWorkers: public Signal::Processing::Workers
{
public:
    TaskWorkers(Processing::ISchedule::ptr schedule, Processing::Bedroom::ptr bedroom);

    void addComputingEngine(ComputingEngine::ptr ce) override;
    void removeComputingEngine(ComputingEngine::ptr ce) override;
    const Engines &workers() const override;
    size_t n_workers() const override;
    const EngineWorkerMap& workers_map() const override;
    DeadEngines clean_dead_workers() override;
    void rethrow_any_worker_exception() override;
    bool remove_all_engines(int timeout) const override;
    bool wait(int timeout) const override;

private:
    void updateWorkers();

public:
    static void test();
};

} // namespace TaskWorker
} // namespace Signal

#endif // TASKWORKERS_H
