#ifndef SIGNAL_PROCESSING_SCHEDULE_H
#define SIGNAL_PROCESSING_SCHEDULE_H

#include "dag.h"
#include "worker.h"
#include "gettask.h"

namespace Signal {
namespace Processing {

class Schedule: public GetTask
{
public:
    Schedule(Dag::Ptr g);

    void wakeup();
    bool is_sleeping();

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    void addComputingEngine(Signal::ComputingEngine::Ptr ce);

    // Throw exception if not found
    void removeComputingEngine(Signal::ComputingEngine::Ptr ce);

    std::vector<Signal::ComputingEngine::Ptr> getComputingEngines() const;

private:
    GetTask::Ptr get_task;

    typedef std::map<Signal::ComputingEngine::Ptr, Worker::Ptr> EngineWorkerMap;
    EngineWorkerMap workers;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULE_H
