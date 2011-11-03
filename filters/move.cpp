#include "move.h"
#include "move.cu.h"

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

    ::moveFilter( chunk.transform_data,
                  df, chunk.minHz(), chunk.maxHz(), chunk.sample_rate, (unsigned long)chunk.chunk_offset );
}

} // namespace Filters
