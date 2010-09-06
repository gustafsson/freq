#include "ridge.h"

namespace Filters
{

void Ridge::
        operator()( Tfr::Chunk& chunk )
{
    float2* p     = chunk.transform_data->getCpuMemory();

    std::vector<float2> q[] = {
        std::vector<float2>(chunk.nSamples()),
        std::vector<float2>(chunk.nSamples())
    };

    unsigned bytes_per_row = sizeof(float2)*chunk.nSamples();

    float2
            *prev_row = &q[0][0],
            *this_row = &q[1][0];

    for (unsigned y=1; y<chunk.nScales()-1; y++)
    {
        float2* t = prev_row;
        prev_row = this_row;
        this_row = t;

        for (unsigned x=0; x<chunk.nSamples(); x++)
        {
            float2 a = p[(y-1)*chunk.nSamples() + x];
            float2 b = p[(y+0)*chunk.nSamples() + x];
            float2 c = p[(y+1)*chunk.nSamples() + x];
            float A = a.x*a.x + a.y*a.y;
            float B = b.x*b.x + b.y*b.y;
            float C = c.x*c.x + c.y*c.y;
            this_row[x] = (B > A  &&  B > C) ? b : make_float2(0,0);
        }

        if (1<y)
            memcpy( p + (y-1)*chunk.nSamples(), prev_row, bytes_per_row );
    }
    memcpy( p + (chunk.nScales()-2)*chunk.nSamples(),
            this_row,
            bytes_per_row );
    memset( p + (chunk.nScales()-1)*chunk.nSamples(), 0, bytes_per_row );
    memset( p                                         , 0, bytes_per_row );
}

} // namespace Filters
