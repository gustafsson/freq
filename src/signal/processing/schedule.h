#ifndef SIGNAL_PROCESSING_SCHEDULE_H
#define SIGNAL_PROCESSING_SCHEDULE_H

#include "dag.h"
#include "worker.h"
#include "workers.h"

namespace Signal {
namespace Processing {

class Schedule
{
public:
    // Should take a GetTask as input rather than a Dag
    Schedule(Dag::Ptr g);

    void wakeup();
    bool is_sleeping();

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    void addComputingEngine(Signal::ComputingEngine::Ptr ce);

    // Throw exception if not found
    void removeComputingEngine(Signal::ComputingEngine::Ptr ce);

    Workers::Ptr getWorkers() const;

private:
    class ScheduleWorkers: public Workers {
    public:
        friend class Schedule;
    };

    Workers::Ptr workers_;
    GetTask::Ptr get_task;

    typedef std::map<Signal::ComputingEngine::Ptr, Worker::Ptr> EngineWorkerMap;
    EngineWorkerMap workers;

    void updateWorkers();
public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULE_H
