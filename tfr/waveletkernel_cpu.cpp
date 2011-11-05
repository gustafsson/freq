
#include <resamplecpu.h>

#include "waveletkerneldef.h"


void wtCompute(
        DataStorage<Tfr::ChunkElement>::Ptr in_waveform_ftp,
        Tfr::ChunkData::Ptr out_wavelet_ftp,
        float fs,
        float /*minHz*/,
        float maxHz,
        unsigned half_sizes,
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

    for (unsigned w_bin=0; w_bin<size.width; w_bin++)
    {
        compute_wavelet_coefficients_elem(
                w_bin,
                in_waveform_ft,
                out_wavelet_ft,
                size.width, size.height,
                first_scale,
                scales_per_octave,
                half_sizes,
                sigma_t0,
                normalization_factor );
    }
}


void wtInverse( Tfr::ChunkData::Ptr in_waveletp, DataStorage<float>::Ptr out_inverse_waveform, DataStorageSize size )
{
    // Multiply the coefficients together and normalize the result
    Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<2>(in_waveletp).ptr();
    float* out = CpuMemoryStorage::WriteAll<1>(out_inverse_waveform).ptr();

    for (unsigned x=0; x<size.width; ++x)
    {
        inverse_elem(
                x, in, out,
                size);
    }
}
