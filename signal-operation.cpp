#include "signal-operation.h"

namespace Signal {

Operation::
Operation(pSource source )
:   _source( source )
{
}

pBuffer Operation::
        readChecked( unsigned firstSample, unsigned numberOfSamples )
{
    pBuffer r = read(first, numberOfSamples);

    if (r->sample_offset > firstSample)
        throw std::runtime_error("read didn't contain firstSample, r->sample_offset > firstSample");

    if (r->sample_offset + r->number_of_samples() <= firstSample)
        throw std::runtime_error("read didn't contain firstSample, r->sample_offset + r->number_of_samples() <= firstSample");
}

pBuffer Operation::
        readFixedLength( unsigned firstSample, unsigned numberOfSamples )
{
    // Try a simple read
    pBuffer p = readChecked(firstSample, numberOfSamples );
    if (p->number_of_samples() == numberOfSamples && p->sample_offset==firstSample)
        return p;

    // Didn't get exact result, prepare new Buffer
    pBuffer r( new Buffer );
    r->sample_offset = firstSample;
    r->sample_rate = p->sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent( numberOfSamples, 1, 1)));
    float* c = q->waveform_data->getCpuMemory();

    // Fill new buffer
    unsigned itr = 0;
    do {
        unsigned o = p->sample_offset - firstSample;
        unsigned l = p->number_of_samples()-o;
        memcpy( c+itr, p->waveform_data->getCpuMemory()+o, l*sizeof(float) );

        itr+=l;

        if (itr<numberOfSamples)
            p = readChecked( firstSample + itr, numberOfSamples - itr );

    } while (itr<numberOfSamples);

    return r;
}


unsigned Operation::
sample_rate()
{
    return _source->sample_rate();
}

unsigned Operation::
number_of_samples()
{
    return _source->number_of_samples();
}

SamplesIntervalDescriptor Operation::
updateInvalidSamples()
{
    Operation* o = dynamic_cast<Operation*>(_source.get());

    if (0!=o)
        _invalid_samples |= o->updateInvalidSamples();

    return _invalid_samples;
}

pSource Operation::first_source(pSource start)
{
    Operation* o = dynamic_cast<Operation*>(start.get());
    if (o)
        return first_source(o->source());

    return start;
}

} // namespace Signal
