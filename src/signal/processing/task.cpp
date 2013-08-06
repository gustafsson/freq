#include "task.h"

#include <boost/foreach.hpp>
#include <QThread>

//#define DEBUGINFO
#define DEBUGINFO if(0)

namespace Signal {
namespace Processing {

Task::
        Task(Step* writeable_step,
             Signal::Processing::Step::Ptr step,
             std::vector<Signal::Processing::Step::Ptr> children,
             Signal::Interval expected_output)
    :
      step_(step),
      children_(children),
      expected_output_(expected_output)
{
    if (writeable_step)
        writeable_step->registerTask (this, expected_output);
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
        run(Signal::ComputingEngine::Ptr ce) volatile
{
    DEBUGINFO TaskTimer tt(boost::format("run %1%") % ReadPtr(this)->expected_output());

    Signal::Operation::Ptr o = write1(step_)->operation (ce);

    if (!o) {
        DEBUGINFO TaskInfo("Oups, this engine does not support this operation");
        return;
    }

    Signal::pBuffer input_buffer = get_input();
    Signal::pBuffer output_buffer = o->process (input_buffer);
    DEBUGINFO TaskInfo("finishing");
    finish(output_buffer);
    DEBUGINFO TaskInfo("finished");
}


Signal::pBuffer Task::
        get_input() volatile
{
    DEBUGINFO TaskTimer tt("get input");
    Signal::Intervals needed;
    Signal::OperationDesc::Ptr operation_desc;

    {
        WritePtr self(this);

        needed = self->expected_output_;
        operation_desc = read1(step_)->operation_desc ();
    }

    Signal::Intervals required_input;
    while (needed) {
        Signal::Interval actual_output;
        Signal::Interval r1 = operation_desc->requiredInterval (needed, &actual_output);
        required_input |= r1;
        EXCEPTION_ASSERT (actual_output & needed); // check for valid 'requiredInterval' by making sure that actual_output doesn't stall needed
        needed -= actual_output;
    }

    // Sum all sources
    std::vector<Step::Ptr> children = ReadPtr(this)->children_;
    std::vector<Signal::pBuffer> buffers;
    buffers.reserve (children.size ());

    Signal::OperationDesc::Extent x = operation_desc->extent ();

    unsigned num_channels = x.number_of_channels.get_value_or (0);
    float sample_rate = x.sample_rate.get_value_or (0.f);
    for (size_t i=0;i<children.size(); ++i)
    {
        Signal::pBuffer b = write1(children[i])->readFixedLengthFromCache(required_input.spannedInterval ());
        if (b) {
            num_channels = std::max(num_channels, b->number_of_channels ());
            sample_rate = std::max(sample_rate, b->sample_rate ());
            buffers.push_back ( b );
        }
    }

    if (0==num_channels || 0.f==sample_rate) {
        // Undefined signal. Shouldn't have created this task.
        if (children.empty ()) {
            EXCEPTION_ASSERT(x.sample_rate.is_initialized ());
            EXCEPTION_ASSERT(x.number_of_channels.is_initialized ());
        }
        EXCEPTION_ASSERT_LESS(0, num_channels);
        EXCEPTION_ASSERT_LESS(0, sample_rate);
    }

    Signal::pBuffer input_buffer(new Signal::Buffer(required_input.spannedInterval (), sample_rate, num_channels));

    BOOST_FOREACH( Signal::pBuffer b, buffers )
    {
        for (unsigned c=0; c<num_channels && c<b->number_of_channels (); ++c)
            *input_buffer->getChannel (c) += *b->getChannel(c);
    }

    return input_buffer;
}


void Task::
        finish(Signal::pBuffer b) volatile
{
    Step::Ptr step = ReadPtr(this)->step_;
    if (step)
        write1(step)->finishTask(&*WritePtr(this), b);
    WritePtr(this)->step_.reset();
}


void Task::
        cancel() volatile
{
    Step::Ptr step = ReadPtr(this)->step_;
    if (step)
        write1(step)->finishTask(&*WritePtr(this), Signal::pBuffer());
    WritePtr(this)->step_.reset();
}


void Task::
        test()
{
    // It should store results of an operation in the cache
    {
        // setup a known signal processing operation (take data from a predefined buffer)
        pBuffer b(new Buffer(Interval(60,70), 40, 7));
        for (unsigned c=0; c<b->number_of_channels (); ++c)
        {
            float *p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
            for (int i=0; i<b->number_of_samples (); ++i)
                p[i] = c + i/(float)b->number_of_samples ();
        }
        Signal::OperationDesc::Ptr od(new BufferSource(b));

        // setup a known signal processing step
        Step::Ptr step (new Step(od));
        std::vector<Step::Ptr> children; // empty
        Signal::Interval expected_output(-10,80);

        // perform a signal processing task
        Task t(write1(step), step, children, expected_output);
        t.run (Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));

        Signal::Interval to_read = Signal::Intervals(expected_output).enlarge (2).spannedInterval ();
        Signal::pBuffer r = write1(step)->readFixedLengthFromCache(to_read);
        EXCEPTION_ASSERT_EQUALS(b->sample_rate (), r->sample_rate ());
        EXCEPTION_ASSERT_EQUALS(b->number_of_channels (), r->number_of_channels ());

        Signal::Buffer expected_r(to_read, 40, 7);
        expected_r += *b;

        EXCEPTION_ASSERT(expected_r == *r);
    }
}


} // namespace Processing
} // namespace Signal
