#include "task.h"

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

    Signal::pBuffer output_buffer = o->process (input_buffer);

    {
        Step::WritePtr step_result(step);
        step_result->finishTask(this, output_buffer);
    }
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

    // Assume that the cache has been accurately setup
    int num_channels = write1(step)->num_channels();
    float fs = write1(step)->sample_rate();
    Signal::pBuffer input_buffer(new Signal::Buffer(required_input.spannedInterval (), fs, num_channels));

    // Sum all sources
    std::vector<Step::Ptr> children = ReadPtr(this)->children_;
    for (size_t i=0;i<children.size(); ++i)
        *input_buffer += *write1(children[i])->readFixedLengthFromCache(required_input.spannedInterval ());

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
        Step::Ptr step (new Step(od, b->number_of_channels (), b->sample_rate ()));
        std::vector<Step::Ptr> children; // empty
        Signal::Interval expected_output(-10,80);

        // perform a signal processing task
        Task t(step, children, expected_output);
        t.run (Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));

        EXCEPTION_ASSERT_EQUALS(b->sample_rate (), write1(step)->sample_rate());
        EXCEPTION_ASSERT_EQUALS(b->number_of_channels (), write1(step)->num_channels());

        Signal::Interval to_read = Signal::Intervals(expected_output).enlarge (2).spannedInterval ();
        Signal::pBuffer r = write1(step)->readFixedLengthFromCache(to_read);
        for (unsigned c=0; c<r->number_of_channels (); ++c)
        {
            float *p = r->getChannel (c)->waveform_data ()->getCpuMemory ();
            for (int i=0; i<r->number_of_samples (); ++i)
            {
                float v = 0;
                Signal::IntervalType k = i + r->getInterval ().first;
                if (b->getInterval ().contains (k))
                {
                    int j = k - b->getInterval ().first;
                    v = c + j/(float)b->number_of_samples ();
                }
                EXCEPTION_ASSERT_EQUALS(p[i], v);
            }
        }
    }
}


} // namespace Processing
} // namespace Signal
