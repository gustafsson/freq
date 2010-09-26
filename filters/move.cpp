#include "move.h"
#include "move.cu.h"

// gpumisc
#include <CudaException.h>

//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;

namespace Filters {

Move::
        Move(float df)
:   _df(df)
{}

void Move::
        operator()( Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("Move");

    float df = _df * chunk.nScales();

    ::moveFilter( chunk.transform_data->getCudaGlobal(),
                  df, chunk.min_hz, chunk.max_hz, (float)chunk.sample_rate, chunk.chunk_offset );

    TIME_FILTER CudaException_ThreadSynchronize();
}

} // namespace Filters
