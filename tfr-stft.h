#ifndef TFRSTFT_H
#define TFRSTFT_H

#include <boost/shared_ptr.hpp>
#include "GpuCpuData.h"

typedef unsigned int cufftHandle; /* from cufft.h */

namespace Tfr {

typedef boost::shared_ptr< GpuCpuData<float2> > pStftChunk;

class Stft
{
public:
    Stft( cudaStream_t stream=0 );
    ~Stft();

    pStftChunk operator()( Signal::pBuffer );
private:
    cudaStream_t    _stream;
    pStftChunk      _intermediate_stft;
    cufftHandle     _fft_single;

    void gc();
};

} // namespace Tfr

#endif // TFRSTFT_H
