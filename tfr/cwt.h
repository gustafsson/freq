#ifndef TFRCWT_H
#define TFRCWT_H

#include "transform.h"
#include "signal/source.h"
#include "stft.h"

namespace Tfr {

class CwtChunk;
class CwtChunkPart;

class Cwt:public Transform
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
    Cwt( float scales_per_octave=40, float wavelet_time_suppport=3, cudaStream_t stream=0 );

    static Cwt& Singleton();
    static pTransform SingletonP();

    virtual pChunk operator()( Signal::pBuffer );
    pChunk computeChunkPart( pChunk ft, unsigned first_scale, unsigned n_scales );

    virtual Signal::pBuffer inverse( pChunk );

    float     get_min_hz(float fs) const;
    void      set_min_hz(float f);
    /// returns the nyquist frequency
    float     get_max_hz(float sample_rate) const { return sample_rate/2.f; }
    unsigned  nScales(float FS) const;
    float     scales_per_octave() const { return _scales_per_octave; }
    void      scales_per_octave( float );
    float     tf_resolution() const { return _tf_resolution; }
    void      tf_resolution( float );
    float     sigma() const;
    float     compute_frequency2( float fs, float normalized_scale ) const;

    /**
      wavelet_time_support is the size of overlapping required in the windowed
      cwt, measured in number of wavelet sigmas. How many samples this
      corresponds to can be computed by calling wavelet_time_support_samples.
      @def 3
      */
    float     wavelet_time_support() const { return _wavelet_time_suppport; }
    void      wavelet_time_support( float value ) { _wavelet_time_suppport = value; }

    /**
      Computes the standard deviation in time and frequency using the tf_resolution value. For a given frequency.
      */
    float     morlet_sigma_t( float sample_rate, float hz ) const;
    float     morlet_sigma_f( float hz ) const;

    /**
      Provided so that clients can compute 'good' overlapping sizes. This gives the number of samples that
      wavelet_std_t corresponds to with a given sample rate, aligned to a multiple of 32 to provide for
      faster inverse calculations later on.
      */
    unsigned  wavelet_time_support_samples( float sample_rate ) const;
    unsigned  wavelet_time_support_samples( float sample_rate, float hz ) const;

    /**
      The Cwt will be computed in chunks who are powers of two. Given sample rate and wavelet_std_t,
      compute a good number of valid samples per chunk.
      */
    unsigned  next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );
    unsigned  prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );

    unsigned        find_bin( unsigned j ) const;
private:
    float           j_to_hz( float sample_rate, unsigned j ) const;
    unsigned        hz_to_j( float sample_rate, float hz ) const;

    Signal::pBuffer inverse( Tfr::CwtChunk* );
    Signal::pBuffer inverse( Tfr::CwtChunkPart* );

    Fft             _fft;
    cudaStream_t    _stream;
    float           _min_hz;
    float           _scales_per_octave;
    float           _tf_resolution;
//    CufftHandleContext _fft_many;

    /**
      Default value: _wavelet_time_suppport=3.
      @see wavelet_time_suppport
      */
    float  _wavelet_time_suppport;
};

} // namespace Tfr

#endif // TFRCWT_H
