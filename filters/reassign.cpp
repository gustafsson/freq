#include "reassign.h"

using namespace Tfr;

namespace Filters
{
void Reassign::
        operator()( Chunk& chunk )
{
    limitedCpu( chunk );
}


void Reassign::
        limitedCpu( Chunk& chunk )
{
    TaskTimer tt("Limited reassign");
    float2* p = chunk.transform_data->getCpuMemory();

    unsigned
            block_size_x = 32,
            block_size_y = 16,
            max_dist = block_size_y/2, // Meaning nothing is allowed to move more than 8 rows away from its origin
            // TODO Q: What action should be taken if it wants to move further? Discard, don't move or clamp?
            W = chunk.nSamples(),
            H = chunk.nScales();

    std::vector<float2> q = std::vector<float2>(block_size_x*block_size_y);
    std::vector<float2> r = std::vector<float2>(block_size_x);
    std::vector<float2> r2 = std::vector<float2>(H);

    unsigned bytes_per_row = block_size_x*sizeof(float2);

    memset(&q[0], 0, bytes_per_row*block_size_y);

    unsigned c = 0;
    unsigned long long d = 0;
    for (unsigned bx=0; bx<W/block_size_x; bx++)
    {
        for (unsigned y=0; y<H; y++)
        {
            if (max_dist <= y)
            {
                unsigned finished_row = (y + max_dist) % block_size_y;
                memcpy(p + (y-max_dist)*W + bx*block_size_x, &q[finished_row * block_size_x], bytes_per_row);
                memset(&q[finished_row * block_size_x], 0, bytes_per_row);
            }

            r2[y] = r[block_size_x-1];
            memcpy(&r[0], p + y*W + bx*block_size_x, bytes_per_row);

            for (unsigned x=0; x<block_size_x; x++)
            {
                float2 a;
                if (0==x && 0==bx)
                    continue;
                else if (0==x)
                    a = r2[y];
                else
                    a = r[x - 1];

                float2 b = r[x];
                float pa = atan2(a.y, a.x);
                float pb = atan2(b.y, b.x);
                float FS = chunk.sample_rate;
                float f = ((pb-pa)*FS)/(2*M_PI);
                unsigned i = chunk.freqAxis().getFrequencyIndex( fabsf(f) );

                if (i >= H)
                    i = H-1;

                if ((i > y ? i-y : y-i) >= max_dist )
                {
                    c ++;
                    d += (i > y ? i-y : y-i);

                    // discard
                    //continue;

                    // clamp
                    //if (i>y)
                    //    i = y + max_dist - 1;
                    //else
                    //    i = y - max_dist + 1;

                    // don't move
                    i = y;
                }


                q[ (i % block_size_y) * block_size_x + x].x += a.x;
                q[ (i % block_size_y) * block_size_x + x].y += a.y;
            }
        }

        for (unsigned y=H; y<H+max_dist; y++)
        {
            unsigned finished_row = (y + max_dist) % block_size_y;
            memcpy(p + (y-max_dist)*W + bx*block_size_x, &q[finished_row * block_size_x], bytes_per_row);
            memset(&q[finished_row * block_size_x], 0, bytes_per_row);
        }
    }

    tt.info("Failed reassignments: %u, %.3g%% of all elements. Average distance: %.3g", c, c*100.f/(W*H), ((double)d)/c);
}


void Reassign::
        naiveCpu( Chunk& chunk )
{
    TaskTimer tt("Naive reassign");
    float2* p = chunk.transform_data->getCpuMemory();

    unsigned
            W = chunk.nSamples(),
            H = chunk.nScales();

    std::vector<float2> q[] = {
        std::vector<float2>(H),
        std::vector<float2>(H)};

    unsigned c = 0;
    unsigned long long d = 0;

    float2
            *prev_q = &q[0][0],
            *this_q = &q[0][0];

    memset(&this_q[0], 0, H*sizeof(float2));

    for (unsigned x=1; x<W; x++)
    {

        float2 *T = prev_q;
        prev_q = this_q;
        this_q = T;

        memset(&this_q[0], 0, H*sizeof(float2));

        for (unsigned y=0; y<H; y++)
        {
            float2 a = p[y*W + x - 1];
            float2 b = p[y*W + x];
            float pa = atan2(a.y, a.x);
            float pb = atan2(b.y, b.x);
            float FS = chunk.sample_rate;
            float f = ((pb-pa)*FS)/(2*M_PI);
            unsigned i = chunk.freqAxis().getFrequencyIndex( fabsf(f) );

            this_q[ i ].x += a.x;
            this_q[ i ].y += a.y;
        }

        for (unsigned y=0; y<H; y++)
            p[y*W + x - 1] = prev_q[y];
    }
    for (unsigned y=0; y<H; y++)
        p[y*W + W - 1] = this_q[y];

    tt.info("Failed reassignments: %u, %.3g%% of all elements. Average distance: %.3g", c, c*100.f/(W*H), ((double)d)/c);
}

}
