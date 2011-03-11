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
    pBuffer buffer( new Buffer(b->sample_offset, chunk->nSamples()*chunk->nScales(), b->sample_rate));

    float2* input = chunk->transform_data->getCpuMemory();
    float* output = buffer->waveform_data()->getCpuMemory();

    Signal::IntervalType N = buffer->number_of_samples();

    float arbitrary_normalization = 1000;
    for(Signal::IntervalType i=0; i<N; ++i)
    {
        output[i] = logf(1+fabsf(input[i].x * input[i].x + input[i].y * input[i].y))/chunk_size();
        output[i] *= arbitrary_normalization;
    }

    pChunk cepstra = stft(buffer);
    TaskInfo("Cepstrum debug. Was %s , returned %s ", b->getInterval().toString().c_str(), cepstra->getInterval().toString().c_str());

    cepstra->freqAxis = freqAxis( cepstra->original_sample_rate );

    return cepstra;

}


FreqAxis Cepstrum::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setQuefrency( FS, chunk_size());
    return fa;
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
