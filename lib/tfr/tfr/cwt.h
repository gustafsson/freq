#ifndef TFRCWT_H
#define TFRCWT_H

#include "transform.h"
#include "signal/buffer.h"
#include "fftimplementation.h"

#include <map>

namespace Tfr {

class CwtChunk;
class CwtChunkPart;

class Cwt:public Transform, public TransformDesc
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

    Number of octaves = 10.107 -> min_hz = 20 Hz at fs = 44100.
    */
    Cwt( float scales_per_octave=20, float wavelet_time_suppport=3, float number_of_octaves=10.107f );

    // Transform
    pChunk operator()( Signal::pMonoBuffer ) override;
    Signal::pMonoBuffer inverse( pChunk ) override;
    const TransformDesc* transformDesc() const override { return this; }

    // TransformDesc
    TransformDesc::ptr copy() const override;
    pTransform createTransform() const override;
    float displayedTimeResolution( float FS, float hz ) const override;
    FreqAxis freqAxis( float FS ) const override;
    //virtual Signal::Interval validLength(Signal::pBuffer buffer);
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const override;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const override;
    bool operator==(const TransformDesc& b) const override;


    float     get_min_hz(float fs) const;
    float     get_min_nf() const;
    float     get_wanted_min_hz(float fs) const;
    void      set_wanted_min_hz(float min_hz, float fs);
    /// returns the nyquist frequency
    float     get_max_hz(float sample_rate) const { return sample_rate/2.f; }
    unsigned  nScales() const;
    unsigned  nBins() const;
    float     scales_per_octave() const { return _scales_per_octave; }
    void      scales_per_octave( float, float fs=0 );
    float     tf_resolution() const { return _tf_resolution; }
    void      tf_resolution( float );
    float     sigma() const;

    /**
      wavelet_time_support is the size of overlapping required in the windowed
      cwt, measured in number of wavelet sigmas. How many samples this
      corresponds to can be computed by calling wavelet_time_support_samples.
      @def 3
      */
    float     wavelet_default_time_support() const { return _wavelet_def_time_suppport; }
    float     wavelet_time_support() const { return _wavelet_time_suppport; }
    void      wavelet_time_support( float value ) { _wavelet_time_suppport = value; _wavelet_def_time_suppport = value; }
    void      wavelet_fast_time_support( float value ) { _wavelet_time_suppport = value; }
    float     wavelet_scale_support() const { return _wavelet_scale_suppport; }
    void      wavelet_scale_support( float value ) { _wavelet_scale_suppport = value; }

    /**
      Computes the standard deviation in time and frequency using the tf_resolution value. For a given frequency.
      */
    float     morlet_sigma_samples( float nf ) const;
    float     morlet_sigma_f( float hz ) const;

    /**
      Provided so that clients can compute 'good' overlapping sizes. This gives the number of samples that
      wavelet_std_t corresponds to with a given sample rate, aligned to a multiple of 32 to provide for
      faster inverse calculations later on.
      */
    unsigned  wavelet_time_support_samples() const;
    unsigned  wavelet_time_support_samples( float nf ) const;

    /**
      The Cwt will be computed in chunks who are powers of two. Given sample rate and wavelet_std_t,
      compute a good number of valid samples per chunk.
      */
    unsigned next_good_size( unsigned current_valid_samples_per_chunk, float /*fs*/) const override {
        return next_good_size(current_valid_samples_per_chunk);
    }
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float /*fs*/) const override {
        return prev_good_size(current_valid_samples_per_chunk);
    }
    unsigned next_good_size( unsigned current_valid_samples_per_chunk ) const;
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk ) const;
    std::string toString() const override;

    void      largest_scales_per_octave( float scales, float last_ok = 0 );
    bool      is_small_enough() const;
    size_t    required_gpu_bytes(unsigned valid_samples_per_chunk) const;

    unsigned        chunk_alignment() const;

private:
    pChunk          computeChunkPart( pChunk ft, unsigned first_scale, unsigned n_scales );

    unsigned        find_bin( unsigned j ) const;
    unsigned        time_support_bin0() const;
    float           j_to_hz( float sample_rate, unsigned j ) const;
    unsigned        hz_to_j( float sample_rate, float hz ) const;
    float           j_to_nf( unsigned j ) const;
    unsigned        nf_to_j( float nf ) const;
    float           hz_to_nf( float sample_rate, float hz ) const;
    float           nf_to_hz( float sample_rate, float nf ) const;
    unsigned        required_length( unsigned current_valid_samples_per_chunk, unsigned&r ) const;
    void            scales_per_octave_internal( float );
    unsigned        chunkpart_alignment(unsigned c) const;

    Signal::pMonoBuffer inverse( Tfr::CwtChunk* );
    Signal::pMonoBuffer inverse( Tfr::CwtChunkPart* );

    float           _number_of_octaves;
    float           _scales_per_octave;
    float           _tf_resolution;
    float           _least_meaningful_fraction_of_r;
    unsigned        _least_meaningful_samples_per_chunk;

    std::map<int, Tfr::FftImplementation::ptr> fft_instances;
    std::set<int> fft_usage; // remove any that wasn't recently used
    Tfr::FftImplementation::ptr fft() const;
    Tfr::FftImplementation::ptr fft(int width); // width=0 -> don't care
    void clearFft();

    /**
      Default value: _wavelet_time_suppport=3.
      @see wavelet_time_suppport
      */
    float _wavelet_time_suppport;
    float _wavelet_def_time_suppport;
    float _wavelet_scale_suppport;
    float _jibberish_normalization;
};

} // namespace Tfr

#endif // TFRCWT_H
