#ifndef SIGNAL_PROCESSING_WORKERS_H
#define SIGNAL_PROCESSING_WORKERS_H

#include "worker.h"
#include "ischedule.h"
#include "signal/computingengine.h"
#include "bedroom.h"

#include <vector>
#include <map>

namespace Signal {
namespace Processing {

class BedroomSignalAdapter;

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
class Workers: public QObject
{
    Q_OBJECT
public:
    typedef shared_state<Workers> ptr;

    // Appended to exceptions created by clean_dead_workers and thrown by rethrow_one_worker_exception
    typedef boost::error_info<struct crashed_engine_tag, Signal::ComputingEngine::ptr> crashed_engine;
    typedef boost::error_info<struct crashed_engine_typename_tag, std::string> crashed_engine_typename;

    typedef std::map<Signal::ComputingEngine::ptr, Worker::ptr> EngineWorkerMap;

    Workers(ISchedule::ptr schedule, Bedroom::ptr bedroom);
    ~Workers();

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    Worker::ptr addComputingEngine(Signal::ComputingEngine::ptr ce);

    /**
     * Prevents the worker for this ComputingEngine to get new work from the
     * scheduler but doesn't kill the thread. Workers keeps a reference to the
     * worker until it has finished.
     *
     * Does nothing if this engine was never added or already removed. An engine
     * will be removed if its worker has finished (or crashed with an exception)
     * and been cleaned by rethrow_any_worker_exception() or clean_dead_workers().
     */
    void removeComputingEngine(Signal::ComputingEngine::ptr ce);

    typedef std::vector<Signal::ComputingEngine::ptr> Engines;
    const Engines& workers() const;
    size_t n_workers() const;
    const EngineWorkerMap& workers_map() const;

    /**
     * Check if any workers has died. This also cleans any dead workers.
     * It is only valid to call this method from the same thread as they were
     * added.
     */
    typedef std::map<Signal::ComputingEngine::ptr, std::exception_ptr > DeadEngines;
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

    bool wait(int timeout=1000);

    static void print(const DeadEngines&);

signals:
    void worker_quit(std::exception_ptr, Signal::ComputingEngine::ptr);

private:
    ISchedule::ptr schedule_;
    BedroomSignalAdapter* notifier_;

    Engines workers_;

    EngineWorkerMap workers_map_;

    void updateWorkers();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERS_H
