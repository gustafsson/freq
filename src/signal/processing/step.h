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

    Signal::SinkSource          cache;
    Signal::Intervals           todo;
    std::vector<boost::shared_ptr<volatile Task> > running_tasks;

    Signal::Intervals           currently_processing() const; // from running_tasks
    Signal::Intervals           out_of_date() const; // todo | currently_processing

    Signal::Operation::Ptr      operation(Signal::ComputingEngine::Ptr);
    Signal::OperationDesc::Ptr  operation_desc() const;
    void                        operation_desc(Signal::OperationDesc::Ptr);

private:
    typedef std::map<Signal::ComputingEngine::WeakPtr, Signal::Operation::Ptr> OperationMap;

    Signal::OperationDesc::Ptr  operation_desc_;
    OperationMap                operations_;

    void                        gc();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_STEP_H
