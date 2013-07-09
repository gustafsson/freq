#ifndef SIGNAL_PROCESSING_TASK_H
#define SIGNAL_PROCESSING_TASK_H

#include "volatileptr.h"

#include "signal/intervals.h"
#include "signal/buffer.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "step.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class Task: public VolatilePtr<Task>
{
public:
    // input_buffer and output_buffer does not need to be allocated beforehand
    Task (Step::Ptr step, std::vector<Step::Ptr> children, Signal::Interval expected_output);

    enum State {
        NotStarted,
        PreparingInput,
        Processing,
        Done
    };

    State                   state() const;
    Signal::Interval        expected_output() const;

    void run(Signal::ComputingEngine::Ptr) volatile;

private:
    Step::Ptr               step_;
    std::vector<Step::Ptr>  children_;
    Signal::Interval        expected_output_;
    State                   state_;

    Signal::pBuffer         get_input() volatile;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TASK_H
