#ifndef TFRCWT_H
#define TFRCWT_H

#include "signal-source.h"
#include "tfr-stft.h"
#include "tfr-chunk.h"

namespace Tfr {


class Cwt
{
public:
    /**
      TODO: Update and clarify
    Cwt initializes asynchronous computation if 0!=stream (inherited
    behaviour from cuda). Also note, from NVIDIA CUDA Programming Guide section
    3.2.6.1, v2.3 Stream:

    "Two commands from different streams cannot run concurrently if either a page-
    locked host memory allocation, a device memory allocation, a device memory set,
    a device ↔ device memory copy, or any CUDA command to stream 0 is called in-
    between them by the host thread."

    That is, a call to Cwt shall never result in any of the operations
    mentioned above, which basically leaves kernel execution and page-locked
    host ↔ device memory copy. For this purpose, a page-locked chunk of memory
    could be allocated beforehand and reserved for the Signal::pBuffer. Previous
    copies with the page-locked chunk is synchronized at beforehand.
    */
    Cwt( float scales_per_octave=40, float wavelet_std_t=0.03f, cudaStream_t stream=0 );

    pChunk operator()( Signal::pBuffer );

    float     min_hz() const { return _min_hz; }
    void      min_hz(float f);
    float     max_hz(unsigned sample_rate) const { return sample_rate/2; }
    float     number_of_octaves( unsigned sample_rate ) const;
    unsigned  nScales(unsigned FS) { return (unsigned)(number_of_octaves(FS) * scales_per_octave()); }
    unsigned  scales_per_octave() const { return _scales_per_octave; }
    void      scales_per_octave( unsigned );

    /**
      wavelet_std_t is the size of overlapping required in the windowed cwt, measured in time units (i.e seconds).
      Cwt does not use this parameter at all, but it is provided here so that clients can compute how much
      overlap they need to provide to get a transform that is continous over window borders.
      */
    float     wavelet_std_t() const { return _wavelet_std_t; }
    void      wavelet_std_t( float value ) { _wavelet_std_t = value; }

    /**
      Provided so that clients can compute 'good' overlapping sizes. This gives the number of samples that
      wavelet_std_t corresponds to with a given sample rate, aligned to a multiple of 32 to provide for
      faster inverse calculations later on.
      */
    unsigned  wavelet_std_samples( unsigned sample_rate ) const;

private:
    Fft             _fft;
    cudaStream_t    _stream;
    float           _min_hz;
    unsigned        _scales_per_octave;

    /**
      Default value: wavelet_std_samples=0.03.
      @see wavelet_std_t
      */
    float  _wavelet_std_t;
};
typedef boost::shared_ptr<Cwt> pCwt;

class CwtSingleton
{
public:
    static pChunk operate( Signal::pBuffer );

    static pCwt instance();
};

} // namespace Tfr

#endif // TFRCWT_H
