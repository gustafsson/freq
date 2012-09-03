#ifndef USE_CUDA
#include <resamplecpu.h>

#include "waveletkerneldef.h"

#include <TaskTimer.h>

#define SQRTLOG2E        1.201122409f
#define PI               3.141592654f

void wtCompute(
        DataStorage<Tfr::ChunkElement>::Ptr in_waveform_ftp,
        Tfr::ChunkData::Ptr out_wavelet_ftp,
        float fs,
        float /*minHz*/,
        float maxHz,
        int half_sizes,
        float scales_per_octave,
        float sigma_t0,
        float normalization_factor )
{
    Tfr::ChunkElement* in_waveform_ft = CpuMemoryStorage::ReadOnly<1>( in_waveform_ftp ).ptr();
    Tfr::ChunkElement* out_wavelet_ft = CpuMemoryStorage::WriteAll<2>( out_wavelet_ftp ).ptr();

    DataStorageSize size = out_wavelet_ftp->size();

//    nyquist = FS/2
//    a = 2 ^ (1/v)
//    aj = a^j
//    hz = fs/2/aj
//    maxHz = fs/2/(a^j)
//    (a^j) = fs/2/maxHz
//    exp(log(a)*j) = fs/2/maxHz
//    j = log(fs/2/maxHz) / log(a)
//    const float log2_a = log2f(2.f) / v = 1.f/v; // a = 2^(1/v)
    float j = (log2f(fs/2) - log2f(maxHz)) * scales_per_octave;
    float first_scale = j;

    j = floor(j+0.5f);

    if (j<0) {
        printf("j = %g, maxHz = %g, fs = %g\n", j, maxHz, fs);
        printf("Invalid argument, maxHz must be less than or equal to fs/2.\n");
        return;
    }

    const float log2_a = 1.f / scales_per_octave; // a = 2^(1/v)

    int nFrequencyBins = size.width, nScales = size.height;
    const int N = nFrequencyBins/2;

    normalization_factor *= sqrt( 4*PI*sigma_t0 );
    normalization_factor *= 2.f/(float)(nFrequencyBins*half_sizes);

    float wscale = 2*PI/nFrequencyBins;
    sigma_t0 *= SQRTLOG2E;

    //for (int w_bin=0; w_bin<N; ++w_bin)
    //        compute_wavelet_coefficients_elem(
    //                w_bin,
    //                in_waveform_ft,
    //                out_wavelet_ft,
    //                size.width, size.height,
    //                first_scale,
    //                scales_per_octave,
    //                sigma_t0,
    //                normalization_factor );

#pragma omp parallel for
    for( int j=0; j<nScales; j++)
    {
        Tfr::ChunkElement waveform_ft;


        // Find period for this thread

        int offset = (nScales-1-j)*nFrequencyBins;
        float aj = exp2f(log2_a * (j + first_scale) ) * wscale;
        for (int w_bin=0; w_bin<N; ++w_bin)
        {
            waveform_ft = in_waveform_ft[w_bin];
            waveform_ft *= normalization_factor;
            if (0==w_bin)
                waveform_ft *= 0.5f;

            float q = (-w_bin*aj + PI)*sigma_t0;

            // Write wavelet coefficient in output matrix
            out_wavelet_ft[offset + w_bin] = waveform_ft * exp2f( -q*q );
        }
        for (int w_bin=N; w_bin<nFrequencyBins; ++w_bin)
            out_wavelet_ft[offset + w_bin] = Tfr::ChunkElement(0,0);
    }
}


void wtInverse( Tfr::ChunkData::Ptr in_waveletp, DataStorage<float>::Ptr out_inverse_waveform, DataStorageSize size )
{
    // Multiply the coefficients together and normalize the result
    Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<2>(in_waveletp).ptr();
    float* out = CpuMemoryStorage::WriteAll<1>(out_inverse_waveform).ptr();

    int w = size.width;
    for (int x=0; x<w; ++x)
    {
        inverse_elem(
                x, in, out,
                size);
    }
}


void wtClamp( Tfr::ChunkData::Ptr in_wtp, size_t sample_offset, Tfr::ChunkData::Ptr out_clamped_wtp )
{
    CpuMemoryReadOnly<Tfr::ChunkElement, 2> in_wt = CpuMemoryStorage::ReadOnly<2>( in_wtp );
    CpuMemoryWriteOnly<Tfr::ChunkElement, 2> out_clamped_wt = CpuMemoryStorage::WriteAll<2>( out_clamped_wtp );

    int h = out_clamped_wt.numberOfElements().height;
    int w = out_clamped_wt.numberOfElements().width;
    for (int y=0; y<h; ++y)
    {
        for (int x=0; x<w; ++x)
        {
            CpuMemoryReadOnly<Tfr::ChunkElement, 2>::Position readp( x + sample_offset, y);
            CpuMemoryWriteOnly<Tfr::ChunkElement, 2>::Position writep( x, y);

            out_clamped_wt.write( writep, in_wt.read( readp ));
        }
    }
}


#endif // USE_CUDA
