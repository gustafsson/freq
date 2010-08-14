#ifndef TFRSTFT_H
#define TFRSTFT_H

#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"
#include "signal-source.h"
#include <vector>

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {


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

typedef boost::shared_ptr< GpuCpuData<float2> > pFftChunk;

/**
Computes the complex Fast Fourier Transform of a Signal::Buffer.
*/
class Fft
{
public:
    Fft( /*cudaStream_t stream=0*/ );
    ~Fft();

    pFftChunk operator()( Signal::pBuffer b ) { return forward(b);}

    pFftChunk forward( Signal::pBuffer );
    pFftChunk backward( Signal::pBuffer );

private:
//    CufftHandleContext _fft_single;
    cudaStream_t _stream;
    std::vector<double> w;
    std::vector<int> ip;
    std::vector<double> q;

    pFftChunk computeWithOoura( Signal::pBuffer buffer, int direction );
    pFftChunk computeWithCufft( Signal::pBuffer buffer, int direction );
};

/**
Computes the Short-Time Fourier Transform, or Windowed Fourier Transform.
@see Stft::operator()
*/
class Stft
{
public:
    Stft( cudaStream_t stream=0 );

    /**
      Signal::pBuffer is used as data structure for output as well as input, this is to simplify the fourier transform which thus can be done inplace.
      The number of elements divided by chunk_size gives the number of windows. pFftChunk will have a width equal to chunk_size and a height
      equal to the number of windows.
      */
    Signal::pBuffer operator()( Signal::pBuffer);

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned chunk_size;
private:

    cudaStream_t    _stream;
};

typedef boost::shared_ptr<Stft> pStft;

class StftSingleton
{
public:
    static pStft instance();
};

} // namespace Tfr

#endif // TFRSTFT_H
