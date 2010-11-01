#include "supersample.h"
#include "tfr/stft.h"

namespace Filters {

Signal::pBuffer SuperSample::
        supersample( Signal::pBuffer b, float requested_sample_rate )
{
    if (b->sample_rate == requested_sample_rate)
        return b;

    float bsample_rate = b->sample_rate;
    float fmultiple = log2f( requested_sample_rate/bsample_rate );
    int multiple = (int)fmultiple;
    if (fmultiple < 0 )
        throw std::invalid_argument("requested_sample_rate must be bigger than "
                                    "b->sample_rate");
    if (fmultiple != multiple)
        throw std::invalid_argument("requested_sample_rate must be an exact "
                                    "multiple of b->sample_rate");

    Tfr::Fft ft;
    Tfr::pChunk chunk = ft( b );
    Tfr::pChunk biggerchunk( new Tfr::StftChunk );
    biggerchunk->min_hz = chunk->min_hz;
    biggerchunk->max_hz = chunk->max_hz;
    biggerchunk->axis_scale = chunk->axis_scale;
    biggerchunk->chunk_offset = chunk->chunk_offset << multiple;
    biggerchunk->first_valid_sample = chunk->first_valid_sample << multiple;
    biggerchunk->n_valid_samples = chunk->n_valid_samples << multiple;
    biggerchunk->order = chunk->order;

    cudaExtent src_sz = chunk->transform_data->getNumberOfElements();
    cudaExtent dest_sz = src_sz;
    dest_sz.width <<= multiple;
    biggerchunk->transform_data.reset( new GpuCpuData<float2>(0, dest_sz) );

    biggerchunk->sample_rate = requested_sample_rate / biggerchunk->nScales();

    float2* src = chunk->transform_data->getCpuMemory();
    float2* dest = biggerchunk->transform_data->getCpuMemory();

    size_t half = src_sz.width/2;

    memcpy( dest, src, half * sizeof(float2) );
    memset( dest + half, 0, (dest_sz.width - src_sz.width) * sizeof(float2) );
    memcpy( dest + (dest_sz.width - half), src + half, half * sizeof(float2) );

    Signal::pBuffer r = ft.inverse( biggerchunk );
    r->sample_rate = requested_sample_rate;
    return r;
}

} // namespace Filters
