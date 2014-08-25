#ifndef SIGNAL_PROCESSING_STEP_H
#define SIGNAL_PROCESSING_STEP_H

#include "shared_state.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "signal/cache.h"

#include <condition_variable>

namespace Signal {
namespace Processing {

class Task;

/**
 * @brief The Step class should keep a cache for a signal processing step
 * (defined by an OpertionDesc).
 *
 * The cache description should contain information about what's out_of_date
 * and what's currently being updated.
 *
 * A crashed signal processing step should behave as a transparent operation.
 */
class Step
{
public:
    typedef shared_state<Step> ptr;
    typedef shared_state<const Step> const_ptr;
    struct shared_state_traits : shared_state_traits_backtrace {
        virtual double timeout () { return 4*shared_state_traits_default::timeout (); }
        virtual double verify_lock_time() { return timeout()/4.0f; }
#if defined SHARED_STATE_NO_TIMEOUT
        typedef shared_state_mutex_notimeout_noshared shared_state_mutex;
#else
        typedef shared_state_mutex_noshared shared_state_mutex;
#endif
    };

    // To be appended to exceptions while using Step
    struct crashed_step_tag {};
    typedef boost::error_info<struct crashed_step_tag, Step::ptr> crashed_step;

    Step(Signal::OperationDesc::ptr operation_desc);

    Signal::OperationDesc::ptr get_crashed() const;
    Signal::Processing::IInvalidator::ptr mark_as_crashed_and_get_invalidator();
    void undie();

    /**
     * @brief deprecateCache should mark which intervals the scheduler should find tasks for.
     * @param deprecated_input If this is Signal::Interval::Interval_ALL the entire cache_ is
     * released and the operation map is cleared.
     * @return when an interval of the input is changed it may affect a larger interval.
     * deprecateCache returns which intervals in the cache that was affected by 'deprecated_input'
     */
    Signal::Intervals           deprecateCache(Signal::Intervals deprecated_input);

    /**
     * @brief purge discards samples from the cache, freeing up memory
     * @param still_needed describes which samples to keep
     * @return how many samples that were released
     */
    size_t                      purge(Signal::Intervals still_needed);

    /**
     * @brief not_started describes which samples stuff might be in the cache or in the middle of being processed
     * The cache isn't updated until a new interval is finished. When deprecateCache is called
     * it affects the currently processing samples as well.
     * @return
     */
    Signal::Intervals           not_started() const; // ~cache->samplesDesc() & ~currently_processing;

    static Signal::OperationDesc::ptr operation_desc(const_ptr step);

    int                         registerTask(Signal::Interval expected_output);
    static void                 finishTask(Step::ptr, int taskid, Signal::pBuffer result);

    /**
     * @brief sleepWhileTasks wait until all created tasks for this step has been finished.
     * @param sleep_ms Sleep indefinitely if sleep_ms < 0.
     * @return true if all tasks where finished within sleep_ms, false otherwise.
     */
    static bool                 sleepWhileTasks(Step::ptr::read_ptr& step, int sleep_ms);
    static bool                 sleepWhileTasks(Step::ptr::read_ptr&& step, int sleep_ms);

    /**
     * @brief cache returns a read-only cache. This can be used to query cache
     * contents but not to update the cache. The cache is modified through
     * purge() and finishTask ()
     * @return
     */
    static shared_state<const Signal::Cache> cache(const_ptr step);

private:
    typedef std::map<int, Signal::Intervals> RunningTaskMap;

    Signal::OperationDesc::ptr  died_;
    shared_state<Signal::Cache> cache_;
    int                         task_counter_ = 0;

    RunningTaskMap              running_tasks;

    Signal::OperationDesc::ptr  operation_desc_;

    mutable std::condition_variable_any wait_for_tasks_;

    std::string                 operation_name() const; // doesn't need lock
    Signal::Intervals           currently_processing() const; // from running_tasks

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
