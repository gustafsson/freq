#include "move.cu.h"


#include <stdio.h>

// gpumisc
#include "cudaUtil.h"
#include "CudaException.h"
#include "cudaPitchedPtrType.h"
#include "cudaglobalstorage.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define M_PIf ((float)(M_PI))

__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate, unsigned sample_offset );

void moveFilter( Tfr::ChunkData::Ptr c, float df, float min_hz, float max_hz, float sample_rate, unsigned sample_offset )
{
    cudaPitchedPtrType<float2> chunk(CudaGlobalStorage::ReadWrite<2>( c ).getCudaPitchedPtr());

    elemSize3_t numElem = chunk.getNumberOfElements();
    dim3 block(256,1,1);
    dim3 grid( int_div_ceil(numElem.x, block.x), 1, 1);

    if(grid.x>65535) {
        printf("Invalid argument, number of floats in complex signal must be less than 65535*256.");
        return;
    }

    //float start = sampleRate/minHz/numElem.width;
    float start = min_hz/2;
    float steplogsize = log(max_hz)-log(min_hz);

    kernel_move<<<grid, block>>>( chunk, df, start, steplogsize, sample_rate, sample_offset );

    CudaException_ThreadSynchronize();
}

__global__ void kernel_move(cudaPitchedPtrType<float2> chunk, float df, float start, float steplogsize, float sample_rate, unsigned sample_offset )
{
    const unsigned
            x = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned nSamples = chunk.getNumberOfElements().x;
    unsigned nFrequencies = chunk.getNumberOfElements().y;
    if( x >= nSamples )
        return;

    for( unsigned fc = 0<df ? 1:nFrequencies ;
                       0<df ? fc<=nFrequencies : fc>0;
                       0<df?fc++:fc--)
    {
        // fc is a counter that is off by one, it goes [1,nFrequencies] or [nFrequencies,1]
        unsigned fi = fc-1; // fi=write index

        float ri = fi + df; // read index
        float ff_read = ri/(float)nFrequencies;
        float ff_write = fi/(float)nFrequencies;

        //float period = start*exp(-ff*steplogsize); // start = sampleRate/minHz/numElem.width

        float hz_read = start*exp(ff_read*steplogsize); // start = min_hz/2
        float hz_write = start*exp(ff_write*steplogsize); // start = min_hz/2

        elemSize3_t readPos;
        readPos.x = x;
        readPos.y = ri;
        readPos.z = 0;
        float2 w = make_float2(0,0);
        if ( chunk.valid(readPos) ) {
            float2 c = chunk.elem(readPos);

            // compute how many periods have elapsed at x for readPos.y
            float time = (sample_offset+x) / sample_rate; // same time for both read and write
            float read_angle = time * hz_read*2*M_PIf;
            float write_angle = time * hz_write*2*M_PIf;
            float phaseAngle = atan2( c.y, c.x );
            float phase = fmodf(read_angle + phaseAngle + 2*M_PIf, 2*M_PIf);
            float f = write_angle;// + phase;

            float amplitude = sqrt(c.x*c.x + c.y*c.y);
            w.x = cos(f)*amplitude;
            w.y = sin(f)*amplitude;
            w.x = c.x;
            w.y = c.y;
        }

        elemSize3_t writePos;
        writePos.x = x;
        writePos.y = fi;
        writePos.z = 0;

        chunk.e(writePos) = w;
    }
}
