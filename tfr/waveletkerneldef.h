#include "waveletkernel.h"

#include "resample.h"

/**
  Well, strictly speaking this doesn't produce the 'true' wavelet coefficients
  but rather the coefficients resulting from inversing the wavelet coefficients,
  still in the fourier domain.

  Each thread computes the scale corresponding to the highest frequency first
  and loops down to the scale corresponding to the lowest frequency.

  TODO see matlab file

  @param in_waveform_ft
  Given input signal in fourier domain.

  @param out_wavelet_ft
  Preallocated output coefficients in fourier domain

  @param numElem
  2D size of out_wavelet_ft. numElem.x is size of in_waveform_ft.
  numElem.y is number of scales.

  @param first_scale
  The first scale to compute, first_scale=0 corresponds to the nyquist
  frequency.

  @param v
  Scales per octave is commonly refered to as 'v' in the wavelet bible.

  @param sigma_t0
  Sigma of the mother gabor wavelet in the time domain. Describes the
  time-frequency resolution ratio.
  */
template<typename T>
inline RESAMPLE_CALL void compute_wavelet_coefficients_elem(
        unsigned w_bin,
        T* in_waveform_ft,
        T* out_wavelet_ft,
        unsigned nFrequencyBins, unsigned nScales, float first_scale, float v, float sigma_t0,
        const float& normalization_factor )
{
    const float pi = 3.141592654f;
    const float
            //w = 2.6515*log2f(w_bin*2*pi/nFrequencyBins); // quasi loglets
            w = w_bin*2*pi/nFrequencyBins;

    T waveform_ft;

    if (w_bin>nFrequencyBins/2)
    {
        // Negative frequencies are defined as 0 and are not stored in in_waveform_ft
#ifdef __CUDACC__
        waveform_ft = make_float2(0,0);
#else
        waveform_ft = T(0,0);
#endif

    }
    else
    {
        waveform_ft = in_waveform_ft[w_bin];
        waveform_ft *= normalization_factor;
        if (0==w_bin)
            waveform_ft *= 0.5f;
    }

    // Find period for this thread
    const float log2_a = 1.f / v; // a = 2^(1/v)

    for( unsigned j=0; j<nScales; j++)
    {
        // Compute the child wavelet
        // a = 2^(1/v)
        // aj = a^j
        // aj = pow(a,j) = exp(log(a)*j)
#ifdef __CUDACC__
        T output = make_float2(0,0);
        if (waveform_ft.x != 0 || waveform_ft.y != 0)
#else
        T output(0,0);
        if (waveform_ft.real() != 0 || waveform_ft.imag() != 0)
#endif
        {
            float aj = exp2f(log2_a * (j + first_scale) );

            {
                // Different scales may have different mother wavelets, kind of
                // That is, different sigma_t0 for different j
                // ff = j / (float)total_nScales
                // float f0 = 2.0f + 35*ff*ff*ff
            }
            float q = (-w*aj + pi)*sigma_t0;
            float phi_star = expf( -q*q );

            output = waveform_ft * phi_star;
        }

        // Find offset for this wavelet coefficient. Writes the scale
        // corresponding to the lowest frequency on the first row of the
        // output matrix
        unsigned offset = (nScales-1-j)*nFrequencyBins;

        // Write wavelet coefficient in output matrix
        out_wavelet_ft[offset + w_bin] = output;
    }
}


template<typename T>
inline RESAMPLE_CALL void inverse_elem( unsigned x, T* in_wavelet, float* out_inverse_waveform, DataStorageSize numElem )
{
    if (x>=numElem.width )
        return;

    float a = 0;

    // no selection
    for (unsigned fi=0; fi<numElem.height; fi++)
    {
        T v = in_wavelet[ x + fi*numElem.width ];
#ifdef __CUDACC__
        a += v.x;
#else
        a += v.real();
#endif
    }

    out_inverse_waveform[x] = a;
}
