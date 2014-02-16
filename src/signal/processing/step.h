#ifndef SIGNAL_PROCESSING_STEP_H
#define SIGNAL_PROCESSING_STEP_H

#include "volatileptr.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "signal/cache.h"

#include <QWaitCondition>

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
class Step: public VolatilePtr<Step>
{
public:
    // To be appended to exceptions while using Step
    typedef boost::error_info<struct crashed_step_tag, Step::Ptr> crashed_step;

    Step(Signal::OperationDesc::Ptr operation_desc);

    Signal::OperationDesc::Ptr  get_crashed() const;
    void                        mark_as_crashed();

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

    Signal::OperationDesc::Ptr  operation_desc() const;

    void                        registerTask(Task* taskid, Signal::Interval expected_output);
    void                        finishTask(Task* taskid, Signal::pBuffer result);
    void                        sleepWhileTasks(int sleep_ms);

    /**
     * @brief readFixedLengthFromCache should read a buffer from the cache.
     * @param I
     * @return If no task has finished yet, a null buffer. Otherwise the data
     *         that is stored in the cache for given interval. Cache misses are
     *         returned as 0 values.
     */
    Signal::pBuffer             readFixedLengthFromCache(Signal::Interval I) const;

private:
    typedef std::map<Task*, Signal::Interval> RunningTaskMap;

    Signal::OperationDesc::Ptr  died_;
    Signal::Cache               cache_;
    Signal::Intervals           not_started_;

    RunningTaskMap              running_tasks;

    Signal::OperationDesc::Ptr  operation_desc_;

    boost::condition_variable_any wait_for_tasks_;

    std::string                 operation_name();
    Signal::Intervals           currently_processing() const; // from running_tasks

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
