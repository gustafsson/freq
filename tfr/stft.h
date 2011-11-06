#ifndef TFRSTFT_H
#define TFRSTFT_H

#include "transform.h"

// std
#include <vector>
#include <complex>

// gpumisc
#include "HasSingleton.h"

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {

enum FftDirection
{
    FftDirection_Forward = -1,
    FftDirection_Backward = 1
};

/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.
*/
class Fft: public Transform, public HasSingleton<Fft, Transform>
{
public:
    Fft( bool computeRedundant=false );
    ~Fft();

    virtual pChunk operator()( Signal::pBuffer b ) { return forward(b); }
    virtual Signal::pBuffer inverse( pChunk c ) { return backward(c); }
    virtual FreqAxis freqAxis( float FS );
    virtual float displayedTimeResolution( float FS, float hz );

    pChunk forward( Signal::pBuffer );
    Signal::pBuffer backward( pChunk );

    /**
      Returns the smallest ok chunk size strictly greater than x that also is
      a multiple of 'multiple'.
      'multiple' must be a power of 2.
      */
    static unsigned sChunkSizeG(unsigned x, unsigned multiple=1);

    /**
      Returns the largest ok chunk size strictly smaller than x that also is
      a multiple of 'multiple'.
      'multiple' must be a power of 2.
      */
    static unsigned lChunkSizeS(unsigned x, unsigned multiple=1);

private:
    friend class Stft;

    bool _compute_redundant;

    void computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    void computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );

    void computeWithOouraR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
    void computeWithCufftR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
    void computeWithOouraC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );
    void computeWithCufftC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );
};

/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.
@see Stft::operator()
*/
class Stft: public Transform, public HasSingleton<Stft,Transform>
{
public:
    Stft();

    /**
      The contents of the input Signal::pBuffer is converted to complex values.
      */
    virtual pChunk operator()( Signal::pBuffer );
    virtual Signal::pBuffer inverse( pChunk );
    virtual Signal::pBuffer inverseWithRedundant( pChunk );
    virtual FreqAxis freqAxis( float FS );
    virtual float displayedTimeResolution( float FS, float hz );

    unsigned chunk_size() { return _window_size; }
    unsigned set_approximate_chunk_size( unsigned preferred_size );

    /// @ Try to use set_approximate_chunk_size(unsigned) unless you need an explicit stft size
    void set_exact_chunk_size( unsigned chunk_size ) { _window_size = chunk_size; }

    /**
        If false (default), operator() will do a real-to-complex transform
        instead of a full complex-to-complex.

        (also known as R2C and C2R transforms are being used instead of C2C
        forward and C2C backward)
    */
    bool compute_redundant() { return _compute_redundant; }
    void compute_redundant(bool);


    void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );


    static unsigned build_performance_statistics(bool writeOutput = false, float size_of_test_signal_in_seconds = 10);

private:
    /**
      @see compute_redundant()
      */
    Tfr::pChunk ChunkWithRedundant(Signal::pBuffer breal);

    static std::vector<unsigned> _ok_chunk_sizes;

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned _window_size;
    bool _compute_redundant;

    void computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    Tfr::pChunk computeWithCufft(Signal::pBuffer);
    Tfr::pChunk computeRedundantWithCufft(Signal::pBuffer);
    Signal::pBuffer inverseWithCufft(Tfr::pChunk);
    Signal::pBuffer inverseRedundantWithCufft(Tfr::pChunk);

    void computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    Tfr::pChunk computeWithOoura(Signal::pBuffer);
    Tfr::pChunk computeRedundantWithOoura(Signal::pBuffer);
    Signal::pBuffer inverseWithOoura(Tfr::pChunk);
    Signal::pBuffer inverseRedundantWithOoura(Tfr::pChunk);
};

class StftChunk: public Chunk
{
public:
    StftChunk(unsigned window_size = -1);
    void setHalfs( unsigned n );
    unsigned halfs( );
    unsigned nActualScales() const;

    virtual unsigned nSamples() const;
    virtual unsigned nScales() const;

    virtual Signal::Interval getInterval() const { return getInversedInterval(); }

    // If 'window_size != (unsigned)-1' then this chunks represents a real signal
    // and doesn't contain redundant coefficients.
    // window_size is transform size (including counting redundant coefficients)
    unsigned window_size;

    unsigned transformSize() const;
private:
    unsigned halfs_n;
};

} // namespace Tfr

#endif // TFRSTFT_H
