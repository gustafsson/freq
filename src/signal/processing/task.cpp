#include "task.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Task::
        Task(Signal::Processing::Step::Ptr step,
             std::vector<Signal::Processing::Step::Ptr> children,
             Signal::Interval expected_output)
    :
      step_(step),
      children_(children),
      expected_output_(expected_output)
{
}


Signal::Interval Task::
        expected_output() const
{
    return expected_output_;
}


void Task::
        run(Signal::ComputingEngine::Ptr ce) volatile
{
    Signal::pBuffer input_buffer = get_input();


    Step::Ptr step = ReadPtr(this)->step_;
    Signal::Operation::Ptr o = write1(step)->operation (ce);

    if (!o)
        return;

    Signal::pBuffer output_buffer = o->process (input_buffer);

    write1(step)->finishTask(this, output_buffer);
}


Signal::pBuffer Task::
        get_input() volatile
{
    Step::Ptr step;
    Signal::Intervals needed;
    Signal::OperationDesc::Ptr operation_desc;

    {
        WritePtr self(this);

        step = self->step_;
        needed = self->expected_output_;
        operation_desc = read1(step)->operation_desc ();
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

    unsigned num_channels = 1; // Because a Signal::Buffer with 0 channels is invalid.
    float sample_rate = 0;
    for (size_t i=0;i<children.size(); ++i)
    {
        Signal::pBuffer b = write1(children[i])->readFixedLengthFromCache(required_input.spannedInterval ());
        if (b) {
            num_channels = std::max(num_channels, b->number_of_channels ());
            sample_rate = std::max(sample_rate, b->sample_rate ());
            buffers.push_back ( b );
        }
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
        Task t(step, children, expected_output);
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
