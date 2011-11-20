#ifndef TFRSTFT_H
#define TFRSTFT_H

#include "transform.h"

// std
#include <vector>
#include <complex>

// gpumisc
#include "HasSingleton.h"

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

    /// Fft::inverse does not normalize the result. To normalize it you have to divide each element with the length of the buffer.
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
    /// Stft::inverse does normalize the result (to the contrary of Fft::inverse)
    virtual Signal::pBuffer inverse( pChunk );
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
    Tfr::pChunk ComputeChunk(Signal::pBuffer b);

    /**
      @see compute_redundant()
      */
    Tfr::pChunk ChunkWithRedundant(Signal::pBuffer breal);
    virtual Signal::pBuffer inverseWithRedundant( pChunk );


    static std::vector<unsigned> _ok_chunk_sizes;

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned _window_size;
    bool _compute_redundant;

    void computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    void computeWithCufft( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize );
    void computeRedundantWithCufft( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n );
    void inverseWithCufft( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );
    void inverseRedundantWithCufft( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n );

    void computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    void computeWithOoura( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize );
    void computeRedundantWithOoura( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n );
    void inverseWithOoura( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );
    void inverseRedundantWithOoura( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n );
};

class StftChunk: public Chunk
{
public:
    StftChunk(unsigned window_size, bool redundant);
    void setHalfs( unsigned n );
    unsigned halfs( );
    unsigned nActualScales() const;

    virtual unsigned nSamples() const;
    virtual unsigned nScales() const;

    virtual Signal::Interval getInterval() const { return getInversedInterval(); }

    /// transformSize = window_size >> halfs_n
    unsigned transformSize() const;
    bool redundant() const { return _redundant; }
    unsigned window_size() const { return _window_size; }

private:
    unsigned _halfs_n;

    /**
      window_size is transform size (including counting redundant coefficients
      whether they are actually present in the data or not)
      */
    unsigned _window_size;

    /// Does this chunk contain redundant data
    bool _redundant;
};

} // namespace Tfr

#endif // TFRSTFT_H
