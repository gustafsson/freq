#include "drawnwaveform.h"

#include "drawnwaveform.cu.h"

#include "neat_math.h"

#include <math.h>

namespace Tfr {


DrawnWaveform::
        DrawnWaveform()
            :
            block_fs(0),
            maxValue(0.0001f)
{}


pChunk DrawnWaveform::
        operator()( Signal::pBuffer b )
{
    float blobsize = blob(b->sample_rate);

    unsigned w = ((unsigned)(b->number_of_samples() / blobsize / drawWaveform_BLOCK_SIZE)) *drawWaveform_BLOCK_SIZE;

    if (0 == w)
        throw std::logic_error("DrawnWaveform::operator() Not enough data");

    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    free /= 2; // Don't even try to get close to use all memory
    // never use more than 64 MB
    if (free > 64<<20)
        free = 64<<20;

    unsigned wmax = free/(drawWaveform_YRESOLUTION*sizeof(float2)*drawWaveform_BLOCK_SIZE ) * drawWaveform_BLOCK_SIZE;
    if (0==wmax)
        wmax = free/(drawWaveform_YRESOLUTION*sizeof(float2));
    if (0==wmax)
        wmax = 1;
    if (wmax < w)
        w = wmax;

    pChunk c(new DrawnWaveformChunk(this->block_fs));
    c->transform_data.reset( new GpuCpuData<float2>(
            0,
            make_uint3(w, drawWaveform_YRESOLUTION, 1),
            GpuCpuVoidData::CudaGlobal));

    unsigned readstop = b->number_of_samples();
    if (b->getInterval().last > signal_length)
    {
        if (b->getInterval().first > signal_length)
            readstop = 0;
        else
            readstop = signal_length - b->getInterval().first;
    }

    ::drawWaveform(
            b->waveform_data(),
            c->transform_data->getCudaGlobal(),
            blobsize,
            readstop,
            maxValue);

    this->block_fs = 0;

    c->chunk_offset = b->sample_offset / blobsize;
    c->freqAxis = freqAxis( b->sample_rate );

    c->first_valid_sample = 0;
    c->n_valid_samples = c->nSamples();

    c->original_sample_rate = b->sample_rate;
    c->sample_rate = b->sample_rate / blobsize;

    return c;
}


Signal::pBuffer DrawnWaveform::
        inverse( pChunk /*chunk*/ )
{
    throw std::logic_error("Not implemented");
}


float DrawnWaveform::
        displayedTimeResolution( float FS, float /*hz*/ )
{
    return .25f/(FS);
}


FreqAxis DrawnWaveform::
        freqAxis( float FS )
{
    FreqAxis a;
    a.setLinear(FS, drawWaveform_YRESOLUTION - 1);
    //a.setLogarithmic(20, FS/2, drawWaveform_YRESOLUTION - 1);

    //a.setLinear(44100, drawWaveform_YRESOLUTION - 1);
    /*a.axis_scale = AxisScale_Linear;
    a.max_frequency_scalar = drawWaveform_YRESOLUTION - 1;
    float max_hz = 1000;
    a.min_hz = 0;
    a.f_step = (1/a.max_frequency_scalar) * (max_hz - a.min_hz);
*/
    return a;
}


float DrawnWaveform::
        blob(float FS)
{
    if (0 == this->block_fs)
        this->block_fs = FS;
    float b = FS / this->block_fs;
    if (b > 1)
        b = (unsigned)(b + 0.5f);
    return b;
}


} // namespace Tfr
