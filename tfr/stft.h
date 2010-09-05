#ifndef TFRSTFT_H
#define TFRSTFT_H

#include "transform.h"
#include <vector>

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {


/**
  CufftHandleContext is used by Cwt but could also be used by Stft and Fft.
  */
class CufftHandleContext {
public:
    CufftHandleContext( cudaStream_t _stream=0 ); // type defaults to cufftPlan1d( CUFFT_C2C )
    ~CufftHandleContext();

    cufftHandle operator()( unsigned elems, unsigned batch_size );

private:
    ThreadChecker _creator_thread;
    cufftHandle _handle;
    cudaStream_t _stream;
    unsigned _elems;
    unsigned _batch_size;

    void destroy();
    void create();
};

// typedef boost::shared_ptr< GpuCpuData<float2> > pFftChunk;

/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.
*/
class Fft: public Transform
{
public:
    Fft( /*cudaStream_t stream=0*/ );
    ~Fft();

    pChunk operator()( Signal::pBuffer b ) { return forward(b); }

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
};

/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.
@see Stft::operator()
*/
class Stft: public Transform
{
public:
    Stft( cudaStream_t stream=0 );

    static Stft& Singleton();
    static pTransform SingletonP();

    /**
      The contents of the input Signal::pBuffer is converted to complex values.
      */
    pChunk operator()( Signal::pBuffer );

    unsigned chunk_size() { return _chunk_size; }
    unsigned set_approximate_chunk_size( unsigned preferred_size );

    static unsigned build_performance_statistics(bool writeOutput = false, float size_of_test_signal = 1);
private:
    cudaStream_t    _stream;

    static std::vector<unsigned> _ok_chunk_sizes;

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned _chunk_size;
};

} // namespace Tfr

#endif // TFRSTFT_H
