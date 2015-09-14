#include <cmath>

//#ifndef USE_CUDA

#include "normalizekernel.h"
#include "cpumemorystorage.h"

#if defined(__GNUC__)
    #include <cmath>
#endif

#define VAL(v) fabsf(v)
#define INVVAL(v) (v)
//#define VAL(v) (v*v)
//#define INVVAL(v) sqrt(v)

// TODO could optimize this by computing the rms more sparsely and interpolate the rms value on a spline, would work really well in cuda as well
void normalizedata(
        DataStorage<float>::ptr data,
        int radius )
{
    unsigned width = data->size().width;

    float* p = CpuMemoryStorage::ReadOnly<1>( data ).ptr();

    int channels = data->size().height*data->size().depth;
#pragma omp parallel for
    for (int c=0; c<channels; ++c)
    {
        double sum = 0.f;
        double N = radius + 1 + radius;
        for (int t=-radius; t<radius; ++t)
        {
            float v = p[radius+t];
            sum += VAL(v);
        }

        for (unsigned x=radius; x<width-radius; ++x)
        {
            float v = p[x+radius];
            sum += VAL(v);

            float invsum = INVVAL(N / sum);

            v = p[x-radius];
            sum -= VAL(v);

            p[x-radius] = p[x]*invsum;
        }
    }
}


// TODO not in use
void normalizedatawindowed(
        DataStorage<float>::ptr data,
        int radius )
{
    unsigned width = data->size().width;

    float* p = CpuMemoryStorage::ReadOnly<1>( data ).ptr();

    int channels = data->size().height*data->size().depth;
#pragma omp parallel for
    for (int c=0; c<channels; ++c)
    {
        for (unsigned x=radius; x<width-radius; ++x)
        {
            float squaresum = 0.f;
            for (int t=-radius; t<=radius; ++t)
            {
                float w = (2.f - fabsf(t/(float)radius));
                float v = w*p[x+t];
                squaresum += v*v;
            }

            float invrms = sqrt((2*radius+1) / squaresum);
            p[x-radius] = p[x]*invrms;
        }
    }
}


// TODO not finished
void normalizeTruncatedMean(
        DataStorage<float>::ptr data,
        int radius, float truncation )
{
    unsigned width = data->size().width;

    float* p = CpuMemoryStorage::ReadOnly<1>( data ).ptr();

    int channels = data->size().height*data->size().depth;
#pragma omp parallel for
    for (int c=0; c<channels; ++c)
    {
        for (unsigned x=radius; x<width-radius; ++x)
        {
            float squaresum = 0.f;
            for (int t=-radius; t<=radius; ++t)
            {
                float w = (2.f - fabsf(t/(float)radius));
                float v = w*p[x+t];
                squaresum += v*v;
            }

            float rms = sqrt((2*radius+1) / squaresum);

            squaresum = 0.f;
            unsigned N = 0;
            for (int t=-radius; t<=radius; ++t)
            {
                float w = (2.f - fabsf(t/(float)radius));
                float v = w*p[x+t];
                if (fabsf(v) < rms*truncation)
                {
                    ++N;
                    squaresum += v*v;
                }
            }

            float invrms = sqrt(N / squaresum);
            p[x-radius] = p[x]*invrms;
        }
    }
}

//#endif // USE_CUDA
