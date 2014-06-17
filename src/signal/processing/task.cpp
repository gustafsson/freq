#include "task.h"

#include "tasktimer.h"
#include "demangle.h"
#include "expectexception.h"
#include "log.h"

#include <boost/foreach.hpp>

//#define TIME_TASK
#define TIME_TASK if(0)

namespace Signal {
namespace Processing {

Task::
        Task(const shared_state<Step>::write_ptr& step,
             Step::ptr stepp,
             std::vector<Step::const_ptr> children,
             Signal::Operation::ptr operation,
             Signal::Interval expected_output,
             Signal::Interval required_input)
    :
      step_(stepp),
      children_(children),
      operation_(operation),
      expected_output_(expected_output),
      required_input_(required_input)
{
    step->registerTask (this, expected_output);
}


Task::
        ~Task()
{
    cancel();
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
            EXCEPTION_ASSERT(i);
            i.read ()->deprecateCache (Signal::Intervals::Intervals_ALL);
        } catch(const std::exception& y) {
            x << unexpected_exception_info(boost::current_exception());
        }

        throw;
      }
}


void Task::
        run_private()
{
    Signal::OperationDesc::ptr od;
    TIME_TASK od = step_.raw ()->operation_desc ();
    TIME_TASK TaskTimer tt(boost::format("Task::run %1%")
                           % od.read ()->toString ().toStdString ());

    Signal::Operation::ptr o = this->operation_;

    Signal::pBuffer input_buffer, output_buffer;

    {
        TIME_TASK TaskTimer tt(boost::format("expect  %s")
                               % expected_output());
        input_buffer = get_input();
    }

    {
        TIME_TASK TaskTimer tt(boost::format("process %s") % input_buffer->getInterval ());
        output_buffer = o->process (input_buffer);
        finish(output_buffer);
    }
}


Signal::pBuffer Task::
        get_input() const
{
    Signal::OperationDesc::ptr operation_desc = step_.raw ()->operation_desc ();

    // Sum all sources
    std::vector<Signal::pBuffer> buffers;
    buffers.reserve (children_.size ());

    Signal::OperationDesc::Extent x = operation_desc.read ()->extent ();

    unsigned num_channels = x.number_of_channels.get_value_or (0);
    float sample_rate = x.sample_rate.get_value_or (0.f);
    for (size_t i=0;i<children_.size(); ++i)
    {
        Signal::pBuffer b = children_[i].read ()->readFixedLengthFromCache(required_input_);
        if (b) {
            num_channels = std::max(num_channels, b->number_of_channels ());
            sample_rate = std::max(sample_rate, b->sample_rate ());
            buffers.push_back ( b );
        }
    }

    if (0==num_channels || 0.f==sample_rate) {
        // Undefined signal. Shouldn't have created this task.
        Log("required_input = %s") % required_input_;
        if (children_.empty ()) {
            EXCEPTION_ASSERT(x.sample_rate.is_initialized ());
            EXCEPTION_ASSERT(x.number_of_channels.is_initialized ());
        } else {
            EXCEPTION_ASSERT_LESS(0u, buffers.size ());
        }
        EXCEPTION_ASSERT_LESS(0u, num_channels);
        EXCEPTION_ASSERT_LESS(0u, sample_rate);
    }

    Signal::pBuffer input_buffer(new Signal::Buffer(required_input_, sample_rate, num_channels));

    BOOST_FOREACH( Signal::pBuffer b, buffers )
    {
        for (unsigned c=0; c<num_channels && c<b->number_of_channels (); ++c)
            *input_buffer->getChannel (c) += *b->getChannel(c);
    }

    return input_buffer;
}


void Task::
        finish(Signal::pBuffer b)
{
    if (step_)
        step_.write ()->finishTask(this, b);
    step_.reset();
}


void Task::
        cancel()
{
    if (step_)
        step_.write ()->finishTask(this, Signal::pBuffer());
    step_.reset();
}

} // namespace Processing
} // namespace Signal

#include "test/randombuffer.h"

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
        Signal::pBuffer r = step.write ()->readFixedLengthFromCache(to_read);
        EXCEPTION_ASSERT_EQUALS(b->sample_rate (), r->sample_rate ());
        EXCEPTION_ASSERT_EQUALS(b->number_of_channels (), r->number_of_channels ());

        Signal::Buffer expected_r(to_read, 40, 7);
        expected_r += *b;

        EXCEPTION_ASSERT(expected_r == *r);
    }
}


} // namespace Processing
} // namespace Signal
