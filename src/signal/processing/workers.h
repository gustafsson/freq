#ifndef SIGNAL_PROCESSING_WORKERS_H
#define SIGNAL_PROCESSING_WORKERS_H

#include "worker.h"
#include "ischedule.h"
#include "signal/computingengine.h"

#include <vector>
#include <map>

namespace Signal {
namespace Processing {

/**
 * @brief The Schedule class should start and stop computing engines as they
 * are added and removed.
 *
 * A started engine is a thread that uses class Worker which queries a GetTask
 * for tasks to work on.
 *
 * It should terminate all threads when it's closed.
 */
class Workers: public QObject, public VolatilePtr<Workers>
{
    Q_OBJECT
public:
    // Appended to exceptions created by clean_dead_workers and thrown by rethrow_one_worker_exception
    typedef boost::error_info<struct crashed_engine, Signal::ComputingEngine::Ptr> crashed_engine_value;

    typedef std::map<Signal::ComputingEngine::Ptr, Worker::Ptr> EngineWorkerMap;

    Workers(ISchedule::Ptr schedule);
    ~Workers();

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    Worker::Ptr addComputingEngine(Signal::ComputingEngine::Ptr ce);

    /**
     * Throw exception if this engine was never added or already removed. The
     * thread can be stopped without calling removeComputingEngine. Call
     * clean_dead_workers() to remove them from the workers() list.
     */
    void removeComputingEngine(Signal::ComputingEngine::Ptr ce);

    typedef std::vector<Signal::ComputingEngine::Ptr> Engines;
    const Engines& workers() const;
    size_t n_workers() const;
    const EngineWorkerMap& workers_map() const;

    /**
     * Check if any workers has died. This also cleans any dead workers.
     * It is only valid to call this method from the same thread as they were
     * added.
     */
    typedef std::map<Signal::ComputingEngine::Ptr, boost::exception_ptr > DeadEngines;
    DeadEngines clean_dead_workers();
    void rethrow_any_worker_exception();

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
    bool terminate_workers(int timeout=1000);

    /**
     * @brief remove_all_engines will ask all workers to not start any new
     * task from now on.
     *
     * Returns true if all threads finished within 'timeout'.
     */
    bool remove_all_engines(int timeout=0) const;

    static void print(const DeadEngines&);

signals:
    void worker_quit(boost::exception_ptr, Signal::ComputingEngine::Ptr);

private slots:
    void worker_quit_slot();

private:
    ISchedule::Ptr schedule_;

    Engines workers_;

    EngineWorkerMap workers_map_;

    void updateWorkers();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERS_H
