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
        typedef shared_state_mutex_noshared shared_state_mutex;
    };

    // To be appended to exceptions while using Step
    typedef boost::error_info<struct crashed_step_tag, Step::ptr> crashed_step;

    Step(Signal::OperationDesc::ptr operation_desc);

    Signal::OperationDesc::ptr get_crashed() const;
    Signal::Processing::IInvalidator::ptr mark_as_crashed_and_get_invalidator();

    /**
     * @brief deprecateCache should mark which intervals the scheduler should find tasks for.
     * @param deprecated_input If this is Signal::Interval::Interval_ALL the entire cache_ is
     * released and the operation map is cleared.
     * @return when an interval of the input is changed it may affect a larger interval.
     * deprecateCache returns which intervals in the cache that was affected by 'deprecated_input'
     */
    Signal::Intervals           deprecateCache(Signal::Intervals deprecated_input);
    Signal::Intervals           not_started() const;
    Signal::Intervals           out_of_date() const; // not_started | currently_processing

    Signal::OperationDesc::ptr  operation_desc() const; // Safe to call without lock

    void                        registerTask(Task* taskid, Signal::Interval expected_output);
    static void                 finishTask(Step::ptr, Task* taskid, Signal::pBuffer result);

    /**
     * @brief sleepWhileTasks wait until all created tasks for this step has been finished.
     * @param sleep_ms Sleep indefinitely if sleep_ms < 0.
     * @return true if all tasks where finished within sleep_ms, false otherwise.
     */
    static bool                 sleepWhileTasks(Step::ptr::read_ptr& step, int sleep_ms);
    static bool                 sleepWhileTasks(Step::ptr::read_ptr&& step, int sleep_ms);

    /**
     * @brief readFixedLengthFromCache should read a buffer from the cache.
     * @param I
     * @return If no task has finished yet, a null buffer. Otherwise the data
     *         that is stored in the cache for given interval. Cache misses are
     *         returned as 0 values.
     */
    static Signal::pBuffer      readFixedLengthFromCache(Step::const_ptr, Signal::Interval I);

private:
    typedef std::map<Task*, Signal::Interval> RunningTaskMap;

    Signal::OperationDesc::ptr  died_;
    shared_state<Signal::Cache> cache_;
    Signal::Intervals           not_started_;

    RunningTaskMap              running_tasks;

    Signal::OperationDesc::ptr  operation_desc_;

    mutable std::condition_variable_any wait_for_tasks_;

    std::string                 operation_name();
    Signal::Intervals           currently_processing() const; // from running_tasks

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
