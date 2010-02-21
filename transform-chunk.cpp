#include "transform-chunk.h"

#include <math.h>

Transform_chunk::Transform_chunk()
:   min_hz(0),
    max_hz(0),
    chunk_offset(0),
    sample_rate(0),
    first_valid_sample(0),
    n_valid_samples(0),
    modified(false)
{}

float2 Transform_chunk::getNearestCoeff( float t, float f )
{
    if (!valid())
        return make_float2(0,0);

    if ( t < 0 ) t = 0;
    unsigned s = (unsigned)(t*sample_rate+.5);
    if ( s >= nSamples() ) s=nSamples()-1;

    unsigned fi = getFrequencyIndex(f);

    return transform_data->getCpuMemoryConst()[ fi*nSamples() + s ];
}

float Transform_chunk::getFrequency( unsigned fi ) const
{
    if (!valid())
        return 0;

    return exp(log(min_hz) + (fi/(float)nFrequencies())*(log(max_hz)-log(min_hz)));
}

unsigned Transform_chunk::getFrequencyIndex( float f ) const
{
    if (f<min_hz) f=min_hz;
    if (f>max_hz) f=max_hz;

    unsigned fi = (unsigned)((log(f)-log(min_hz))/(log(max_hz)-log(min_hz))*nFrequencies());
    if (fi>nFrequencies()) fi = nFrequencies()-1;

    return fi;
}

pWaveform_chunk Transform_chunk::computeInverse( pTransform_chunk chunk, cudaStream_t stream ) {
    cudaExtent sz = make_cudaExtent( chunk->n_valid_samples, 1, 1);

    pWaveform_chunk r( new Waveform_chunk());
    r->sample_offset = chunk->chunk_offset + chunk->first_valid_sample;
    r->sample_rate = chunk->sample_rate;
    r->waveform_data.reset( new GpuCpuData<float>(0, sz, GpuCpuVoidData::CudaGlobal) );

    float4 area = make_float4(
            _t1 * _original_waveform->sample_rate() - r->sample_offset,
            _f1 * nScales(),
            _t2 * _original_waveform->sample_rate() - r->sample_offset,
            _f2 * nScales());
    {
        TaskTimer tt(__FUNCTION__);

        // summarize them all
        ::wtInverse( chunk->transform_data->getCudaGlobal().ptr() + chunk->first_valid_sample,
                     r->waveform_data->getCudaGlobal().ptr(),
                     chunk->transform_data->getNumberOfElements(),
                     area,
                     chunk->n_valid_samples,
                     stream );

        CudaException_ThreadSynchronize();
    }

/*    {
        TaskTimer tt("inverse corollary");

        size_t n = r->waveform_data->getNumberOfElements1D();
        float* data = r->waveform_data->getCpuMemory();
        pWaveform_chunk originalChunk = _original_waveform->getChunk(chunk->sample_offset, chunk->nSamples(), _channel);
        float* orgdata = originalChunk->waveform_data->getCpuMemory();

        double sum = 0, orgsum=0;
        for (size_t i=0; i<n; i++) {
            sum += fabsf(data[i]);
        }
        for (size_t i=0; i<n; i++) {
            orgsum += fabsf(orgdata[i]);
        }
        float scale = orgsum/sum;
        for (size_t i=0; i<n; i++)
            data[i] *= scale;
        tt.info("scales %g, %g, %g", sum, orgsum, scale);

        r->writeFile("outtest.wav");
    }*/
    return r;
}
