#include "fftimplementation.h"
#include "neat_math.h"

#ifdef USE_OPENCL
#include "fftclfft.h"
#elif defined(USE_CUDA)
#include "fftcufft.h"
#else
#include "fftooura.h"
#endif

#if defined(USE_CUDA) && !defined(USE_CUFFT)
#define USE_CUFFT
#endif


#include <boost/make_shared.hpp>

using namespace boost;

namespace Tfr {

shared_ptr<FftImplementation> FftImplementation::
        newInstance()
{
#ifdef USE_CUFFT
    return make_shared<FftCufft>(); // Gpu, Cuda
#elif defined(USE_OPENCL)
    return make_shared<FftClFft>(); // Gpu, OpenCL
#else
    return make_shared<FftOoura>(); // Cpu
#endif
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
