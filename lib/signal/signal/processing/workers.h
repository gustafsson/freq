#ifndef SIGNAL_PROCESSING_WORKERS_H
#define SIGNAL_PROCESSING_WORKERS_H

#include "signal/processing/ischedule.h"
#include "signal/computingengine.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/worker.h"

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
 *
 * It should wake up sleeping workers when any work is done to see if they can
 * help out on what's left.
 */
class Workers
{
public:
    typedef shared_state<Workers> ptr;
    typedef std::vector<Signal::ComputingEngine::ptr> Engines;
    typedef std::map<Signal::ComputingEngine::ptr, std::exception_ptr > DeadEngines;
    typedef std::map<Signal::ComputingEngine::ptr, Worker::ptr> EngineWorkerMap;

    // Appended to exceptions created by clean_dead_workers and thrown by rethrow_one_worker_exception
    typedef boost::error_info<struct crashed_engine_tag, Signal::ComputingEngine::ptr> crashed_engine;
    typedef boost::error_info<struct crashed_engine_typename_tag, std::string> crashed_engine_typename;

    Workers(Processing::ISchedule::ptr schedule, Processing::Bedroom::ptr bedroom);
    virtual ~Workers() {}

    // Throw exception if already added.
    // This will spawn a new worker thread for this computing engine.
    virtual void addComputingEngine(Signal::ComputingEngine::ptr ce) = 0;

    /**
     * Prevents the worker for this ComputingEngine to get new work from the
     * scheduler but doesn't kill the thread. Workers keeps a reference to the
     * worker until it has finished.
     *
     * Does nothing if this engine was never added or already removed. An engine
     * will be removed if its worker has finished (or crashed with an exception)
     * and been cleaned by rethrow_any_worker_exception() or clean_dead_workers().
     */
    virtual void removeComputingEngine(Signal::ComputingEngine::ptr ce) = 0;

    virtual const Engines& workers() const = 0;
    virtual size_t n_workers() const = 0;
    virtual const EngineWorkerMap& workers_map() const = 0;

    /**
     * Check if any workers has died. This also cleans any dead workers.
     * It is only valid to call this method from the same thread as they were
     * added.
     */
    virtual DeadEngines clean_dead_workers() = 0;
    virtual void rethrow_any_worker_exception() = 0;

    /**
     * @brief remove_all_engines will ask all workers to not start any new
     * task from now on.
     *
     * Returns true if all threads finished within 'timeout'.
     */
    virtual bool remove_all_engines(int timeout=0) const = 0;

    virtual bool wait(int timeout=1000) const = 0;

    static void print(const DeadEngines&);

protected:
    Engines workers_;
    EngineWorkerMap workers_map_;

    ISchedule::ptr schedule_;
    Bedroom::ptr bedroom_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERS_H
