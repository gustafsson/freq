#ifndef USE_CUDA

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#if defined(_WIN32) && defined(FREQAXIS_CALL)
// defined in precompiled header
#undef FREQAXIS_CALL
#endif

#include "resamplecpu.h"

#include <complex>

template<>
float ConverterAmplitude::
operator()( std::complex<float> w, DataPos const& /*dataPosition*/ )
{
    // slightly faster than sqrtf(f) unless '--use_fast_math' is specified
    // to nvcc
    // return f*rsqrtf(f);
    return abs(w);
}


#include "blockkerneldef.inc"

// that's it, blockkerneldef contains the definitions
#endif // USE_CUDA
