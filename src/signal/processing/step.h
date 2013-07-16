#ifndef SIGNAL_PROCESSING_STEP_H
#define SIGNAL_PROCESSING_STEP_H

#include "volatileptr.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "signal/sinksource.h"

namespace Signal {
namespace Processing {

class Task;

class Step: public VolatilePtr<Step>
{
public:
    Step(Signal::OperationDesc::Ptr operation_desc);

    /**
     * @brief deprecateCache should mark which intervals the scheduler should find tasks for.
     * @param deprecated_input
     * @return which intervals in the cache that was affected by 'deprecated_input'
     */
    Signal::Intervals           deprecateCache(Signal::Intervals deprecated_input);
    void                        setInvalid(Signal::Intervals invalid); // implicitly validates ~invalid
    Signal::Intervals           not_started() const;
    Signal::Intervals           out_of_date() const; // not_started | currently_processing

    Signal::Operation::Ptr      operation(Signal::ComputingEngine::Ptr);
    Signal::OperationDesc::Ptr  operation_desc() const;

    void                        registerTask(volatile Task*, Signal::Interval expected_output);
    void                        finishTask(volatile Task*, Signal::pBuffer result);

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
    typedef std::map<volatile Task*, Signal::Interval> RunningTaskMap;

    Signal::SinkSource::Ptr     cache_;
    Signal::Intervals           not_started_;

    RunningTaskMap              running_tasks;

    Signal::OperationDesc::Ptr  operation_desc_;
    OperationMap                operations_;

    Signal::Intervals           currently_processing() const; // from running_tasks
    void                        gc();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
