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
    Step(Signal::OperationDesc::Ptr operation_desc, float sample_rate, int num_channels);

    void                        deprecateCache(Signal::Intervals deprecated);
    void                        setInvalid(Signal::Intervals invalid); // implicitly validates ~invalid
    Signal::Intervals           not_started() const;
    Signal::Intervals           out_of_date() const; // not_started | currently_processing

    Signal::Operation::Ptr      operation(Signal::ComputingEngine::Ptr);
    Signal::OperationDesc::Ptr  operation_desc() const;

    void                        registerTask(volatile Task*, Signal::Interval expected_output);
    void                        finishTask(volatile Task*, Signal::pBuffer result);

    Signal::pBuffer             readFixedLengthFromCache(Signal::Interval I);
    float                       sample_rate();
    unsigned                    num_channels();

private:
    typedef std::map<Signal::ComputingEngine::WeakPtr, Signal::Operation::Ptr> OperationMap;
    typedef std::map<volatile Task*, Signal::Interval> RunningTaskMap;

    Signal::SinkSource          cache_;
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
