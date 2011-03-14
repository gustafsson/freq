#ifndef TFRSTFT_H
#define TFRSTFT_H

#include "transform.h"
#include <vector>
#include "HasSingleton.h"

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {


/**
  CufftHandleContext is used by Cwt but could also be used by Stft and Fft.
  */
class CufftHandleContext {
public:
    CufftHandleContext( cudaStream_t _stream=0, unsigned type=-1); // type defaults to CUFFT_C2C
    ~CufftHandleContext();

    CufftHandleContext( const CufftHandleContext& b );
    CufftHandleContext& operator=( const CufftHandleContext& b );

    cufftHandle operator()( unsigned elems, unsigned batch_size );

    void setType(unsigned type);

private:
    ThreadChecker _creator_thread;
    cufftHandle _handle;
    cudaStream_t _stream;
    unsigned _type;
    unsigned _elems;
    unsigned _batch_size;

    void destroy();
    void create();
};

// typedef boost::shared_ptr< GpuCpuData<float2> > pFftChunk;

/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.
*/
class Fft: public Transform, public HasSingleton<Fft, Transform>
{
public:
    Fft( /*cudaStream_t stream=0*/ );
    ~Fft();

    virtual pChunk operator()( Signal::pBuffer b ) { return forward(b); }
    virtual Signal::pBuffer inverse( pChunk c ) { return backward(c); }
    virtual FreqAxis freqAxis( float FS );

    pChunk forward( Signal::pBuffer );
    Signal::pBuffer backward( pChunk );

private:
//    CufftHandleContext _fft_single;
    cudaStream_t _stream;
    std::vector<double> w; // used by Ooura
    std::vector<int> ip;
    std::vector<double> q;

    void computeWithOoura( GpuCpuData<float2>& input, GpuCpuData<float2>& output, int direction );
    void computeWithCufft( GpuCpuData<float2>& input, GpuCpuData<float2>& output, int direction );

    void computeWithCufftR2C( GpuCpuData<float>& input, GpuCpuData<float2>& output );
    void computeWithCufftC2R( GpuCpuData<float2>& input, GpuCpuData<float>& output );
};

/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.
@see Stft::operator()
*/
class Stft: public Transform, public HasSingleton<Stft,Transform>
{
public:
    Stft( cudaStream_t stream=0 );

    /**
      The contents of the input Signal::pBuffer is converted to complex values.
      */
    virtual pChunk operator()( Signal::pBuffer );
    virtual Signal::pBuffer inverse( pChunk ) { throw std::logic_error("Not implemented"); }
    virtual FreqAxis freqAxis( float FS );

    unsigned chunk_size() { return _window_size; }
    unsigned set_approximate_chunk_size( unsigned preferred_size );

    /// @ Try to use set_approximate_chunk_size(unsigned) unless you need an explicit stft size
    void set_exact_chunk_size( unsigned chunk_size ) { _window_size = chunk_size; }

    /**
        If false (default), operator() will do a real-to-complex transform
        instead of a full complex-to-complex.
    */
    bool compute_redundant() { return _compute_redundant; }
    void compute_redundant(bool);


    static unsigned build_performance_statistics(bool writeOutput = false, float size_of_test_signal_in_seconds = 10);

private:
    Tfr::pChunk ChunkWithRedundant(Signal::pBuffer breal);

    cudaStream_t    _stream;
    CufftHandleContext _handle_ctx;

    static std::vector<unsigned> _ok_chunk_sizes;

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned _window_size;
    bool _compute_redundant;
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
