#ifndef SIGNAL_PROCESSING_TASK_H
#define SIGNAL_PROCESSING_TASK_H

#include "volatileptr.h"

#include "signal/intervals.h"
#include "signal/buffer.h"
#include "signal/computingengine.h"
#include "signal/operation.h"
#include "step.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Task class should store results of an operation in the cache.
 *
 * If the Task fails, the section of the cache that was supposed to be filled
 * by this Task should be invalidated.
 */
class Task: public VolatilePtr<Task>
{
public:
    // input_buffer and output_buffer does not need to be allocated beforehand
    Task (Step* writeable_step, Step::Ptr step, std::vector<Step::Ptr> children, Signal::Interval expected_output);
    ~Task();

    Signal::Interval        expected_output() const;

    void run(Signal::ComputingEngine::Ptr) volatile;

private:
    Step::Ptr               step_;
    std::vector<Step::Ptr>  children_;
    Signal::Interval        expected_output_;

    Signal::pBuffer         get_input() volatile;
    void                    finish(Signal::pBuffer) volatile;
    void                    cancel() volatile;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TASK_H
