#include "signal-sink.h"
#include "string.h"

namespace Signal {

Sink::Sink()
:   _expected_samples_left( 0 )
{}

unsigned Sink::expected_samples_left()
{
    return _expected_samples_left;
}

void Sink::expected_samples_left(unsigned expected_samples_left)
{
    _expected_samples_left = expected_samples_left;
}

/**
  code snipped left behind:

  @param t target
  @param s source
  */
void merge_buffers(Signal::pBuffer t, Signal::pBuffer s)
{
    unsigned out_offs = (s->sample_offset > t->sample_offset) ? s->sample_offset - t->sample_offset : 0;
    unsigned in_offs =  (t->sample_offset > s->sample_offset) ? t->sample_offset - s->sample_offset : 0;
    unsigned count = s->waveform_data->getNumberOfElements().width;
    if (count>in_offs)
        count -= in_offs;
    else
        return;

    memcpy( t->waveform_data->getCpuMemory() + out_offs,
            s->waveform_data->getCpuMemory() + in_offs,
            count*sizeof(float) );
}

} // namespace Signal
