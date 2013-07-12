#ifndef SIGNAL_PROCESSING_SCHEDULE_H
#define SIGNAL_PROCESSING_SCHEDULE_H

#include "worker.h"
#include "workers.h"
#include "gettask.h"
#include "signal/computingengine.h"
#include "volatileptr.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Schedule class should start and stop computing engines as they
 * are added and removed.
 *
 * A started engine uses class Worker which queries a GetTask for tasks to work
 * on.
 */
class Schedule: public VolatilePtr<Schedule>
{
public:
    Schedule(GetTask::Ptr get_task);

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    void addComputingEngine(Signal::ComputingEngine::Ptr ce);

    // Throw exception if not found
    void removeComputingEngine(Signal::ComputingEngine::Ptr ce);

    typedef std::vector<Signal::ComputingEngine::Ptr> Engines;
    const Engines& getWorkers() const;

    // Check if any workers has died. This also cleans any dead workers.
    typedef std::map<Signal::ComputingEngine::Ptr, std::pair<const std::type_info*,std::string> > DeadEngines;
    DeadEngines clean_dead_workers();

private:
    GetTask::Ptr get_task;

    Engines workers_;

    typedef std::map<Signal::ComputingEngine::Ptr, Worker::Ptr> EngineWorkerMap;
    EngineWorkerMap workers;

    void updateWorkers();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULE_H
