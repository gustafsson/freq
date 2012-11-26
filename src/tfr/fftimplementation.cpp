#include "fftimplementation.h"
#include "neat_math.h"

#include "fftooura.h"
#include "fftclfft.h"
#include "fftcufft.h"

#if defined(USE_CUDA) && !defined(USE_CUFFT)
#define USE_CUFFT
#endif

namespace Tfr {

FftImplementation& FftImplementation::
        Singleton()
{
    // TODO can't use statics
#ifdef USE_CUFFT
    static FftCufft fft;
#elif defined(USE_OPENCL)
    static FftClFft fft;
#else
    static FftOoura fft;
#endif
    return fft;
}


unsigned FftImplementation::
        lChunkSizeS(unsigned x, unsigned)
{
    return lpo2s(x);
}


unsigned FftImplementation::
        sChunkSizeG(unsigned x, unsigned)
{
    return spo2g(x);
}

} // namespace Tfr
