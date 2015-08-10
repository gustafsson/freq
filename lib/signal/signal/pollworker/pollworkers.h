#ifndef SIGNAL_POLLWORKER_WORKERS_H
#define SIGNAL_POLLWORKER_WORKERS_H

#include "pollworker.h"
#include "signal/processing/ischedule.h"
#include "signal/computingengine.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/workers.h"

#include <vector>
#include <map>

namespace Signal {
namespace Processing {
class BedroomSignalAdapter;
}
namespace PollWorker {


/**
 * @brief The Schedule class should start and stop computing engines as they
 * are added and removed.
 *
 * A started engine is a thread that uses class Worker which queries a GetTask
 * for tasks to work on.
 *
 * It should terminate all threads when it's closed.
 *
 * It should wake up sleeping workers when any work is done to see if they can
 * help out on what's left.
 */
class PollWorkers: public QObject, public Signal::Processing::Workers
{
    Q_OBJECT
public:
    typedef std::map<Signal::ComputingEngine::ptr, PollWorker::ptr> EngineWorkerMap;

    PollWorkers(Signal::Processing::ISchedule::ptr schedule, Signal::Processing::Bedroom::ptr bedroom);
    ~PollWorkers();

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    void addComputingEngine(Signal::ComputingEngine::ptr ce) override;

    /**
     * Prevents the worker for this ComputingEngine to get new work from the
     * scheduler but doesn't kill the thread. Workers keeps a reference to the
     * worker until it has finished.
     *
     * Does nothing if this engine was never added or already removed. An engine
     * will be removed if its worker has finished (or crashed with an exception)
     * and been cleaned by rethrow_any_worker_exception() or clean_dead_workers().
     */
    void removeComputingEngine(Signal::ComputingEngine::ptr ce) override;

    const Engines& workers() const override;
    size_t n_workers() const override;
    const EngineWorkerMap& workers_map() const;

    /**
     * Check if any workers has died. This also cleans any dead workers.
     * It is only valid to call this method from the same thread as they were
     * added.
     */
    DeadEngines clean_dead_workers() override;
    void rethrow_any_worker_exception() override;

    /**
     * @brief terminate_workers terminates all worker threads and doesn't
     * return until all threads have been terminated.
     *
     * Use of this function is discouraged because it doesn't allow threads
     * to clean up any resources nor to release any locks.
     *
     * The thread might not terminate until it activates the OS scheduler by
     * entering a lock or going to sleep.
     *
     * Returns true if all threads were terminated within 'timeout'.
     */
    bool terminate_workers(int timeout=1000) override;

    /**
     * @brief remove_all_engines will ask all workers to not start any new
     * task from now on.
     *
     * Returns true if all threads finished within 'timeout'.
     */
    bool remove_all_engines(int timeout=0) const override;

    bool wait(int timeout=1000) override;

signals:
    void worker_quit(std::exception_ptr, Signal::ComputingEngine::ptr);

private:
    Signal::Processing::ISchedule::ptr schedule_;
    Signal::Processing::BedroomSignalAdapter* notifier_;

    Engines workers_;

    EngineWorkerMap workers_map_;

    void updateWorkers();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_POLLWORKER_WORKERS_H
