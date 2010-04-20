#ifndef TFRSTFT_H
#define TFRSTFT_H

#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"
#include "signal-source.h"

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {

typedef boost::shared_ptr< GpuCpuData<float2> > pFftChunk;

class Fft
{
public:
    Fft( cudaStream_t stream=0 );
    ~Fft();

    pFftChunk operator()( Signal::pBuffer );

private:
    cudaStream_t    _stream;
    pFftChunk       _intermediate_fft;
    cufftHandle     _fft_single;

    void gc();
};

class Stft
{
    Stft( cudaStream_t stream=0 );

    /**
      Signal::pBuffer is used as data structure for output as well as input, this is to simplify the fourier transform which thus can be done inplace.
      The number of elements divided by chunk_size gives the number of windows. pFftChunk will have a width equal to chunk_size and a height
      equal to the number of windows.
      */
    Signal::pBuffer operator()( Signal::pBuffer );

    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    unsigned chunk_size;
private:

    cudaStream_t    _stream;
};


} // namespace Tfr

#endif // TFRSTFT_H
