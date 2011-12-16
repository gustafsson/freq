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
    FftDirection_Inverse = 1
};


class StftChunk;


/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.

TODO remove HasSingleton

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
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );
    virtual std::string toString();

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
    std::vector<float> w;
    std::vector<int> ip;
    friend class Stft;

    bool _compute_redundant;

    void computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    void computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
    void computeWithClFft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );

    void computeWithOouraR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
    void computeWithCufftR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
    void computeWithClFftR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
    void computeWithOouraC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );
    void computeWithCufftC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );
    void computeWithClFftC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );
};

/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.

TODO remove HasSingleton

@see Stft::operator()
*/
class Stft: public Transform, public HasSingleton<Stft,Transform>
{
public:
    enum WindowType
    {
        WindowType_Rectangular,
        WindowType_Hann,
        WindowType_Hamming,
        WindowType_Tukey,
        WindowType_Cosine,
        WindowType_Lanczos,
        WindowType_Triangular,
        WindowType_Gaussian,
        WindowType_BarlettHann,
        WindowType_Blackman,
        WindowType_Nuttail,
        WindowType_BlackmanHarris,
        WindowType_BlackmanNuttail,
        WindowType_FlatTop,
        WindowType_NumberOfWindowTypes
    };

    Stft();

    /**
      The contents of the input Signal::pBuffer is converted to complex values.
      */
    virtual pChunk operator()( Signal::pBuffer );
    /// Stft::inverse does normalize the result (to the contrary of Fft::inverse)
    virtual Signal::pBuffer inverse( pChunk );
    virtual FreqAxis freqAxis( float FS );
    virtual float displayedTimeResolution( float FS, float hz );
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );
    virtual std::string toString();

    unsigned increment();
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


    float overlap() { return _overlap; }
    WindowType windowType() { return _window_type; }
    std::string windowTypeName() { return windowTypeName(_window_type); }
    static std::string windowTypeName(WindowType);
    void setWindow(WindowType type, float overlap);


    void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );


    /**
      Different windows are more sutiable for applying the window on the inverse as well.
      */
    bool applyWindowOnInverse(WindowType);

    static unsigned build_performance_statistics(bool writeOutput = false, float size_of_test_signal_in_seconds = 10);

private:
    Tfr::pChunk ComputeChunk(DataStorage<float>::Ptr inputbuffer);

    /**
      @see compute_redundant()
      */
    Tfr::pChunk ChunkWithRedundant(DataStorage<float>::Ptr inputbuffer);
    virtual Signal::pBuffer inverseWithRedundant( pChunk );


    static std::vector<unsigned> _ok_chunk_sizes;

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned _window_size;
    bool _compute_redundant;
    float _overlap;
    WindowType _window_type;

    /**
      prepareWindow applies the window function to some data, using '_window_type' and '_overlap'.
      Will not pad the data with zeros and thus all input data will only be used if it fits
      the overlap function exactly on the sample.
      */
    DataStorage<float>::Ptr prepareWindow( DataStorage<float>::Ptr );
    DataStorage<float>::Ptr reduceWindow( DataStorage<float>::Ptr windowedSignal, const StftChunk* c );

    template<WindowType>
    void prepareWindowKernel( DataStorage<float>::Ptr in, DataStorage<float>::Ptr out );

    template<WindowType>
    void reduceWindowKernel( DataStorage<float>::Ptr in, DataStorage<float>::Ptr out, const StftChunk* c );

    template<WindowType>
    float computeWindowValue( float p );

    void computeWithCufft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction );
    void computeWithCufft( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize );
    void inverseWithCufft( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );

    void computeWithClFft( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction );
    void computeWithClFft( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize );
    void inverseWithClFft( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );

    void computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction );
    void computeWithOoura( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize );
    void inverseWithOoura( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );
};

class StftChunk: public Chunk
{
public:
    StftChunk(unsigned window_size, Stft::WindowType window_type, unsigned increment, bool redundant);
    void setHalfs( unsigned n );
    unsigned halfs( );
    unsigned nActualScales() const;

    virtual unsigned nSamples() const;
    virtual unsigned nScales() const;

    /// transformSize = window_size >> halfs_n
    unsigned transformSize() const;
    bool redundant() const { return _redundant; }
    unsigned window_size() const { return _window_size; }
    Stft::WindowType window_type() const { return _window_type; }
    unsigned increment() const { return _increment; }

    Signal::Interval getCoveredInterval() const;

private:
    Stft::WindowType _window_type;
    unsigned _increment;

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
