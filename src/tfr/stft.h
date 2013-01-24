#ifndef TFRSTFT_H
#define TFRSTFT_H

#include "transform.h"
#include "chunk.h"
#include "fftimplementation.h"
#include "stftdesc.h"
#include "complexbuffer.h"

// std
#include <vector>
#include <complex>


namespace Tfr {


class StftChunk;


/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.
*/
class SaweDll Fft: public Transform, public TransformDesc
{
public:
    Fft( bool computeRedundant=false );
    ~Fft();


    // Implementing Transform
    virtual const TransformDesc* transformDesc() const { return this; }
    virtual pChunk operator()( Signal::pMonoBuffer b ) { return forward(b); }

    /// Fft::inverse does not normalize the result. To normalize it you have to divide each element with the length of the buffer.
    virtual Signal::pMonoBuffer inverse( pChunk c ) { return backward(c); }


    // Implementing TransformParams
    virtual pTransform createTransform() const;
    virtual float displayedTimeResolution( float FS, float hz ) const;
    virtual FreqAxis freqAxis( float FS ) const;
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual std::string toString() const;
    virtual bool operator==(const TransformDesc&) const;


    pChunk forward( Signal::pMonoBuffer );
    Signal::pMonoBuffer backward( pChunk );

    unsigned lChunkSizeS(unsigned x, unsigned multiple=1);
    unsigned sChunkSizeG(unsigned x, unsigned multiple=1);

private:
    bool _compute_redundant;
    FftImplementation::Ptr fft;
};


/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.

@see Stft::operator()
*/
class SaweDll Stft: public Transform
{
public:
    Stft(const StftDesc&s = StftDesc());
    Stft(const Stft&);

    const StftDesc& desc() const { return p; }
    virtual const TransformDesc* transformDesc() const { return &p; }

    /**
      The contents of the input Signal::pBuffer is converted to complex values.
      */
    virtual pChunk operator()( Signal::pMonoBuffer );
    /// Stft::inverse does normalize the result (to the contrary of Fft::inverse)
    virtual Signal::pMonoBuffer inverse( pChunk );

    void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );

    static unsigned build_performance_statistics(bool writeOutput = false, float size_of_test_signal_in_seconds = 10);

    Tfr::ComplexBuffer::Ptr inverseKeepComplex( pChunk chunk );

private:
    const StftDesc p;
    FftImplementation::Ptr fft;

    Tfr::pChunk ComputeChunk(DataStorage<float>::Ptr inputbuffer);

    /**
      @see compute_redundant()
      */
    Tfr::pChunk ChunkWithRedundant(DataStorage<float>::Ptr inputbuffer);
    virtual Signal::pMonoBuffer inverseWithRedundant( pChunk );


    static std::vector<unsigned> _ok_chunk_sizes;


    /**
      prepareWindow applies the window function to some data, using '_window_type' and '_overlap'.
      Will not pad the data with zeros and thus all input data will only be used if it fits
      the overlap function exactly on the sample.
      */
    DataStorage<float>::Ptr prepareWindow( DataStorage<float>::Ptr );
    template<typename T>
    typename DataStorage<T>::Ptr reduceWindow( boost::shared_ptr<DataStorage<T> > windowedSignal, const StftChunk* c );

    template<StftDesc::WindowType>
    void prepareWindowKernel( DataStorage<float>::Ptr in, DataStorage<float>::Ptr out );

    template<StftDesc::WindowType, typename T>
    void reduceWindowKernel( boost::shared_ptr<DataStorage<T> > in, typename DataStorage<T>::Ptr out, const StftChunk* c );

    template<StftDesc::WindowType>
    float computeWindowValue( float p );
};

class StftChunk: public Chunk
{
public:
    StftChunk(unsigned window_size, StftDesc::WindowType window_type, unsigned increment, bool redundant);
    void setHalfs( unsigned n );
    unsigned halfs( );
    // make clear how these are related to the number of data samples in transform_data
    unsigned nActualScales() const;

    virtual unsigned nSamples() const;
    virtual unsigned nScales() const;

    /// transformSize = window_size >> halfs_n
    unsigned transformSize() const;
    bool redundant() const { return _redundant; }
    unsigned window_size() const { return _window_size; }
    StftDesc::WindowType window_type() const { return _window_type; }
    unsigned increment() const { return _increment; }

    Signal::Interval getCoveredInterval() const;

private:
    StftDesc::WindowType _window_type;
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
