#include "cepstrum.h"
#include "stft.h"

using namespace Signal;
namespace Tfr {

Cepstrum::Cepstrum()
{
}


pChunk Cepstrum::
        operator()( pBuffer b )
{
    Stft stft = Stft::Singleton();
    pChunk chunk = stft(b);
    pBuffer buffer( new Buffer(b->sample_offset, b->number_of_samples()/2, b->sample_rate/2));

    float2* input = chunk->transform_data->getCpuMemory();
    float* output = buffer->waveform_data()->getCpuMemory();

    Signal::IntervalType N = buffer->number_of_samples();

    for(Signal::IntervalType i=0; i<N; ++i)
    {
        output[i] = fabsf(input[i].x * input[i].x + input[i].y * input[i].y);
    }

    stft.set_exact_chunk_size(stft.chunk_size()/2);
    pChunk cepstra = stft(buffer);

    return cepstra;
}


Signal::pBuffer Cepstrum::
        inverse( pChunk )
{
    throw std::logic_error("Not implemented");
}


unsigned Cepstrum::chunk_size()
{
    return Stft::Singleton().chunk_size();
}


} // namespace Tfr
