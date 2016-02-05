#ifdef USE_CUDA
#include "reassign.h"
#include "reassign.cu.h"
#include "tfr/chunk.h"

// gpumisc
#include "computationkernel.h"

#include <vector>
#include <string.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif

//#define TIME_FILTER
#define TIME_FILTER if(0)

using namespace Tfr;

namespace Filters
{
void Reassign::
        operator()( ChunkAndInverse& chunk )
{
    limitedCpu( *chunk.chunk );
}


void Reassign::
        limitedCpu( Chunk& chunk )
{
    TaskTimer tt("Limited reassign");
    std::complex<float>* p = chunk.transform_data->getCpuMemory();

    unsigned
            block_size_x = 32,
            block_size_y = 16,
            max_dist = block_size_y/2, // Meaning nothing is allowed to move more than 8 rows away from its origin
            // TODO Q: What action should be taken if it wants to move further? Discard, don't move or clamp?
            W = chunk.nSamples(),
            H = chunk.nScales();

    std::vector<std::complex<float> > q = std::vector<std::complex<float> >(block_size_x*block_size_y);
    std::vector<std::complex<float> > r = std::vector<std::complex<float> >(block_size_x);
    std::vector<std::complex<float> > r2 = std::vector<std::complex<float> >(H);

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
                std::complex<float> a;
                if (0==x && 0==bx)
                    continue;
                else if (0==x)
                    a = r2[y];
                else
                    a = r[x - 1];

                std::complex<float> b = r[x];
                float pa = atan2(a.imag(), a.real());
                float pb = atan2(b.imag(), b.real());
                float FS = chunk.sample_rate;
                float f = ((pb-pa)*FS)/(2*M_PI);
                unsigned i = chunk.freqAxis.getFrequencyIndex( fabsf(f) );

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


                q[ (i % block_size_y) * block_size_x + x] += a;
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
    std::complex<float>* p = chunk.transform_data->getCpuMemory();

    unsigned
            W = chunk.nSamples(),
            H = chunk.nScales();

    std::vector<std::complex<float> > q[] = {
        std::vector<std::complex<float> >(H),
        std::vector<std::complex<float> >(H)};

    unsigned c = 0;
    unsigned long long d = 0;

    std::complex<float>
            *prev_q = &q[0][0],
            *this_q = &q[0][0];

    memset(&this_q[0], 0, H*sizeof(float2));

    for (unsigned x=1; x<W; x++)
    {

        std::complex<float> *T = prev_q;
        prev_q = this_q;
        this_q = T;

        memset(&this_q[0], 0, H*sizeof(float2));

        for (unsigned y=0; y<H; y++)
        {
            std::complex<float> a = p[y*W + x - 1];
            std::complex<float> b = p[y*W + x];
            float pa = atan2(a.imag(), a.real());
            float pb = atan2(b.imag(), b.real());
            float FS = chunk.sample_rate;
            float f = ((pb-pa)*FS)/(2*M_PI);
            unsigned i = chunk.freqAxis.getFrequencyIndex( fabsf(f) );

            this_q[ i ] += a;
        }

        for (unsigned y=0; y<H; y++)
            p[y*W + x - 1] = prev_q[y];
    }
    for (unsigned y=0; y<H; y++)
        p[y*W + W - 1] = this_q[y];

    tt.info("Failed reassignments: %u, %.3g%% of all elements. Average distance: %.3g", c, c*100.f/(W*H), ((double)d)/c);
}


void Reassign::
        brokenGpu(Tfr::Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("ReassignFilter");

    for (unsigned reassignLoop=0;reassignLoop<1;reassignLoop++)
    {
        ::reassignFilter( chunk.transform_data,
                      chunk.minHz(), chunk.maxHz(), chunk.sample_rate );
    }

    TIME_FILTER ComputationSynchronize();
}


//////////// Tonalize

void Tonalize::
        operator()( ChunkAndInverse& chunk )
{
    brokenGpu(*chunk.chunk);
}


void Tonalize::
        brokenGpu(Tfr::Chunk& chunk )
{
    TIME_FILTER TaskTimer tt("TonalizeFilter");

    ::tonalizeFilter( chunk.transform_data,
                  chunk.minHz(), chunk.maxHz(), chunk.sample_rate );

    TIME_FILTER ComputationSynchronize();
}

}
#else
int USE_CUDA_Reassign;
#endif
