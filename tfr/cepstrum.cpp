#include "cepstrum.h"
#include "stft.h"

#include "TaskTimer.h"
#include "neat_math.h"

using namespace Signal;
namespace Tfr {

Cepstrum::Cepstrum()
{
}


pChunk Cepstrum::
        operator()( pBuffer b )
{
    TaskTimer tt("Cepstrum");
    Stft stft = Stft::Singleton();
    stft.compute_redundant( true );
    pChunk chunk = stft(b);

    unsigned windows = chunk->nSamples();
    unsigned window_size = chunk->nScales();
    pBuffer buffer( new Buffer(b->sample_offset, windows*window_size, b->sample_rate));

    ChunkElement* input = chunk->transform_data->getCpuMemory();
    float* output = buffer->waveform_data()->getCpuMemory();

    Signal::IntervalType N = buffer->number_of_samples();
    
    float arbitrary_normalization = 1;
    float normalization = arbitrary_normalization * 1.f/chunk_size();

    for(Signal::IntervalType i=0; i<N; ++i)
    {
        ChunkElement& p = input[i];
        output[i] = logf( 1 + norm(p) ) * normalization;
    }

    pChunk cepstra = stft(buffer);
    TaskInfo("Cepstrum debug. Was %s , returned %s ", 
        b->getInterval().toString().c_str(), 
        cepstra->getInterval().toString().c_str());

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


float Cepstrum::
        displayedTimeResolution( float FS, float hz )
{
    return Stft::Singleton().displayedTimeResolution( FS, hz );
}


unsigned Cepstrum::
        next_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    if (current_valid_samples_per_chunk<chunk_size())
        return chunk_size();

    return spo2g(align_up(current_valid_samples_per_chunk, chunk_size())/chunk_size())*chunk_size();
}


unsigned Cepstrum::
        prev_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    if (current_valid_samples_per_chunk<2*chunk_size())
        return chunk_size();

    return lpo2s(align_up(current_valid_samples_per_chunk, chunk_size())/chunk_size())*chunk_size();
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
