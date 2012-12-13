#ifndef USE_CUDA

#include <resamplecpu.h>

#include "brushpaintkerneldef.h"

// that's it, brushpaintkerneldef contains the definitions

void Gauss::
        test()
{
    Gauss g(ResamplePos(-1.1, 20), ResamplePos(1.5, 1.5));
    double s = 0;
    double dx = .1, dy = .1;

    for (double y=10; y<30; y+=dy)
        for (double x=-10; x<10; x+=dx)
            s += g.gauss_value(x, y)*dx*dy;

    EXCEPTION_ASSERT_FUZZYEQUALS(1-s, 2.35354e-08, 1e-12);
}

#endif // USE_CUDA
