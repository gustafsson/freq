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
    pBuffer buffer( new Buffer(chunk->chunk_offset, chunk->nSamples()*chunk->nScales(), chunk->original_sample_rate));

    float2* input = chunk->transform_data->getCpuMemory();
    float* output = buffer->waveform_data()->getCpuMemory();

    Signal::IntervalType N = buffer->number_of_samples();

    for(Signal::IntervalType i=0; i<N; ++i)
    {
        output[i] = logf(1+fabsf(input[i].x * input[i].x + input[i].y * input[i].y))/chunk_size();
    }

    pChunk cepstra = stft(buffer);
    TaskInfo("Cepstrum debug. Was %s , returned %s ", b->getInterval().toString().c_str(), cepstra->getInterval().toString().c_str());

    cepstra->axis_scale = AxisScale_Quefrency;
    cepstra->min_hz = 2*cepstra->original_sample_rate/chunk_size();

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
