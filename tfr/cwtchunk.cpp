#include "cwtchunk.h"
#include "tfr/wavelet.cu.h"

namespace Tfr {

pChunk CwtChunkPart::
        cleanChunk() const
{
    const Chunk* c = this;

    pChunk clamped( new CwtChunk );
    clamped->transform_data.reset( new GpuCpuData<float2>(0, make_cudaExtent( c->n_valid_samples, c->nScales(), c->nChannels() )));

    ::wtClamp( c->transform_data->getCudaGlobal(), c->first_valid_sample, clamped->transform_data->getCudaGlobal() );
    clamped->max_hz = c->max_hz;
    clamped->min_hz = c->min_hz;
    clamped->chunk_offset = c->chunk_offset + c->first_valid_sample;
    clamped->sample_rate = c->sample_rate;
    clamped->first_valid_sample = 0;
    clamped->n_valid_samples = c->n_valid_samples;

    return clamped;
}


Signal::Interval CwtChunkPart::
        getInterval() const
{
    return Signal::Interval(
        (chunk_offset + first_valid_sample)*(original_sample_rate/sample_rate),
        (chunk_offset + first_valid_sample + n_valid_samples)*(original_sample_rate/sample_rate)
    );
}


Signal::Interval CwtChunk::
        getInterval() const
{
    return Signal::Interval(
        (chunk_offset + first_valid_sample)*(original_sample_rate/sample_rate),
        (chunk_offset + first_valid_sample + n_valid_samples)*(original_sample_rate/sample_rate)
    );
}

} // namespace Tfr
