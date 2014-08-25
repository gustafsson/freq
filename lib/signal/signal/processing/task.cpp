#include "task.h"

#include "tasktimer.h"
#include "demangle.h"
#include "expectexception.h"
#include "log.h"

#include <boost/foreach.hpp>

#include <algorithm>

//#define TIME_TASK
#define TIME_TASK if(0)

namespace Signal {
namespace Processing {

Task::Task()
    :
        task_id_(0)
{}


Task::
        Task(const shared_state<Step>::write_ptr& step,
             Step::ptr stepp,
             std::vector<Step::const_ptr> children,
             Signal::Operation::ptr operation,
             Signal::Interval expected_output,
             Signal::Interval required_input)
    :
      task_id_(step->registerTask (expected_output)),
      step_(stepp),
      children_(children),
      operation_(operation),
      expected_output_(expected_output),
      required_input_(required_input)
{
}


Task::
        ~Task()
{
    try
    {
        cancel();
    } catch (...) {
        TaskInfo(boost::format("~Task %p\n%s")
                 % ((void*)this)
                 % boost::current_exception_diagnostic_information ());
    }
}


Task& Task::
        operator=(Task&& b)
{
    std::swap(task_id_, b.task_id_);
    std::swap(step_, b.step_);
    std::swap(children_, b.children_);
    std::swap(operation_, b.operation_);
    std::swap(expected_output_, b.expected_output_);
    std::swap(required_input_, b.required_input_);
    return *this;
}


Task::operator bool() const
{
    return (bool)step_;
}


Signal::Interval Task::
        expected_output() const
{
    return expected_output_;
}


void Task::
        run()
{
    try
      {
        run_private();
      }
    catch (const boost::exception& x)
      {
        // Append info to be used at the catch site
        x   << Step::crashed_step(step_)
            << Task::crashed_expected_output(expected_output_);

        try {
            Signal::Processing::IInvalidator::ptr i = step_.write ()->mark_as_crashed_and_get_invalidator();
            if (i)
                i->deprecateCache (Signal::Intervals::Intervals_ALL);
        } catch(const std::exception& y) {
            x << unexpected_exception_info(boost::current_exception());
        }

        throw;
      }
}


void Task::
        run_private()
{
    Signal::OperationDesc::ptr od = Step::operation_desc (step_);
    if (!od)
    {
        cancel ();
        return;
    }

    TIME_TASK TaskTimer tt(boost::format("Task::run %1%")
                           % od.raw ()->toString ().toStdString ());

    Signal::Operation::ptr o = this->operation_;

    Signal::pBuffer input_buffer, output_buffer;

    {
        TIME_TASK TaskTimer tt(boost::format("expect  %s")
                               % expected_output());
        input_buffer = get_input();
        if (!input_buffer)
        {
            cancel();
            return;
        }
    }

    {
        TIME_TASK TaskTimer tt(boost::format("process %s")
                               % input_buffer->getInterval ());
        output_buffer = o->process (input_buffer);
        if (!output_buffer)
        {
            cancel();
            return;
        }
        finish(output_buffer);
    }
}


Signal::pBuffer Task::
        get_input() const
{
    Signal::OperationDesc::ptr operation_desc = Step::operation_desc (step_);

    // Sum all sources
    std::vector<Signal::pBuffer> buffers;
    buffers.reserve (children_.size ());

    Signal::OperationDesc::Extent x = operation_desc.read ()->extent ();

    unsigned num_channels = x.number_of_channels.get_value_or (0);
    float sample_rate = x.sample_rate.get_value_or (0.f);
    for (size_t i=0;i<children_.size(); ++i)
    {
        auto cache = Step::cache (children_[i]).read();
        if (!cache->contains(required_input_))
        {
            // The cache has been invalidated since this task was created, abort task.
            return Signal::pBuffer();
        }
        Signal::pBuffer b = cache->read(required_input_);
        num_channels = std::max(num_channels, b->number_of_channels ());
        sample_rate = std::max(sample_rate, b->sample_rate ());
        buffers.push_back ( b );
    }

    if (buffers.size () == 1)
        return buffers[0];

    Signal::pBuffer input_buffer(new Signal::Buffer(required_input_, sample_rate, num_channels));

    for ( Signal::pBuffer b : buffers )
    {
        for (unsigned c=0; c<num_channels && c<b->number_of_channels (); ++c)
            *input_buffer->getChannel (c) += *b->getChannel(c);
    }

    return input_buffer;
}


void Task::
        finish(Signal::pBuffer b)
{
    if (b)
        EXCEPTION_ASSERT_EQUALS(expected_output_, b->getInterval ());

    if (step_)
    {
        Step::finishTask(step_, task_id_, b);
        step_.reset();
    }
}


void Task::
        cancel()
{
    finish(Signal::pBuffer());
}

} // namespace Processing
} // namespace Signal

#include "test/randombuffer.h"
#include "signal/buffersource.h"

namespace Signal {
namespace Processing {

void Task::
        test()
{
    // It should store results of an operation in the cache
    {
        // setup a known signal processing operation (take data from a predefined buffer)
        pBuffer b = Test::RandomBuffer::randomBuffer (Interval(60,70), 40, 7);
        Signal::OperationDesc::ptr od(new BufferSource(b));

        // setup a known signal processing step
        Step::ptr step (new Step(od));
        std::vector<Step::const_ptr> children; // empty
        Signal::Interval expected_output(-10,80);
        Signal::Interval required_input;
        Signal::Operation::ptr o;
        {
            Signal::ComputingEngine::ptr c(new Signal::ComputingCpu);
            auto r = od.read ();
            required_input = r->requiredInterval(expected_output, 0);
            o = r->createOperation (c.get ());
        }

        // perform a signal processing task
        Task t(step.write (), step, children, o, expected_output, required_input);
        t.run ();

        Signal::Interval to_read = Signal::Intervals(expected_output).enlarge (2).spannedInterval ();
        Signal::pBuffer r = Step::cache (step)->read(to_read);
        EXCEPTION_ASSERT_EQUALS(b->sample_rate (), r->sample_rate ());
        EXCEPTION_ASSERT_EQUALS(b->number_of_channels (), r->number_of_channels ());

        Signal::Buffer expected_r(to_read, 40, 7);
        expected_r += *b;

        EXCEPTION_ASSERT(expected_r == *r);
    }
}


} // namespace Processing
} // namespace Signal
