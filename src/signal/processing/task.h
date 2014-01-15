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
    // To be appended to exceptions while using Task
    typedef boost::error_info<struct crashed_expected_output_tag, Signal::Interval> crashed_expected_output;

    // input_buffer and output_buffer does not need to be allocated beforehand
    Task (Step* writeable_step, Step::Ptr step, std::vector<Step::Ptr> children, Signal::Interval expected_output, Signal::Interval required_input);
    ~Task();

    Signal::Interval        expected_output() const;

    virtual void run(Signal::ComputingEngine::Ptr);

private:
    Step::Ptr               step_;
    std::vector<Step::Ptr>  children_;
    Signal::Interval        expected_output_;
    Signal::Interval        required_input_;

    void                    run_private(Signal::ComputingEngine::Ptr);
    Signal::pBuffer         get_input() const;
    void                    finish(Signal::pBuffer);
    void                    cancel();

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TASK_H
