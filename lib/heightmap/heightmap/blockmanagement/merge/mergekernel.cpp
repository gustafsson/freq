#include "resamplecpu.h"

#include <complex>

#include "resample.h"
#include "operate.h"

#include "mergekernel.h"
#include "neat_math.h"

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif

extern "C"
void blockMerge( BlockData::ptr inBlock,
                 BlockData::ptr outBlock,
                 ResampleArea in_area,
                 ResampleArea out_area)
{
    resample2d_plain<NoConverter<float> >
            (inBlock, outBlock, in_area, out_area);
}
