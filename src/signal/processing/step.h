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
    // It's not obvious from this constructor that 'Step' depends on 'Task'
    Step(Signal::OperationDesc::Ptr operation_desc, int num_channels, float sample_rate);

    Signal::Intervals           not_started;
    Signal::Intervals           out_of_date() const; // not_started | currently_processing

    Signal::Operation::Ptr      operation(Signal::ComputingEngine::Ptr);
    Signal::OperationDesc::Ptr  operation_desc() const;

    void                        registerTask(volatile Task*);
    void                        finishTask(volatile Task*, Signal::pBuffer result);

    Signal::pBuffer             readFixedLengthFromCache(Signal::Interval I);
    float                       sample_rate();
    unsigned                    num_channels();

private:
    typedef std::map<Signal::ComputingEngine::WeakPtr, Signal::Operation::Ptr> OperationMap;

    Signal::SinkSource          cache_;
    std::set<volatile Task* >   running_tasks;

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
