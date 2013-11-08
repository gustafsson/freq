#ifndef SIGNAL_PROCESSING_STEP_H
#define SIGNAL_PROCESSING_STEP_H

#include "volatileptr.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "signal/sinksource.h"

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
 */
class Step: public VolatilePtr<Step>
{
public:
    Step(Signal::OperationDesc::Ptr operation_desc);

    /**
     * @brief deprecateCache should mark which intervals the scheduler should find tasks for.
     * @param deprecated_input If this is Signal::Interval::Interval_ALL the entire cache_ is released.
     * @return when the an interval of the input is changed it may affect a larger interval.
     * deprecateCache returns which intervals in the cache that was affected by 'deprecated_input'
     */
    Signal::Intervals           deprecateCache(Signal::Intervals deprecated_input);
    Signal::Intervals           not_started() const;
    Signal::Intervals           out_of_date() const; // not_started | currently_processing

    Signal::Operation::Ptr      operation(Signal::ComputingEngine::Ptr);
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
    Signal::pBuffer             readFixedLengthFromCache(Signal::Interval I);

private:
    typedef std::map<Signal::ComputingEngine::WeakPtr, Signal::Operation::Ptr> OperationMap;
    typedef std::map<Task*, Signal::Interval> RunningTaskMap;

    Signal::SinkSource::Ptr     cache_;
    Signal::Intervals           not_started_;

    RunningTaskMap              running_tasks;

    Signal::OperationDesc::Ptr  operation_desc_;
    OperationMap                operations_;

    QWaitCondition              wait_for_tasks_;

    std::string                 operation_name();
    Signal::Intervals           currently_processing() const; // from running_tasks
    void                        gc();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
