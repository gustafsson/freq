/*#include "watershead.cu.h"
#include <resample.cu.h>
#include <operate.cu.h>

// Watershed algortihm for one convex region
// (might work for slightly non-convex regions, but that is not guaranteed)

#define INTERMEDIATE_OUTSIDE 0
#define INTERMEDIATE_BORDER 1
#define INTERMEDIATE_INSIDE 2


class Xforward
{
public:
    Xforward(unsigned middle, unsigned sz)
        :   middle(middle),
            sz(sz)
    {}


    template<typename Reader>
    __device__ float operator()( uint2 const& p, Reader& reader )
    {
        if (p.x<middle)
            return 0;

        uint2 q = p;
        float v1, v2, v3;
        v2 = reader(q);
        q.x++;
        v3 = reader(q);
        q.x++;
        while (q.x < sz)
        {
            v1 = v2;
            v2 = v3;
            v3 = reader(q);
            if (v2 < v1 && v2 <= v3)
                return INTERMEDIATE_BORDER;
            q.x++;
        }
        if (q.x >= sz)
            return 2;
        float v = reader(p);
        float phase1 = DefaultFetcher<float, ConverterPhase>()( p, reader );
        float phase2 = DefaultFetcher<float, ConverterPhase>()( make_uint2(p.x, p.y+1), reader );
        float phasediff = phase2 - phase1;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;
        float s = 1000;
        float k = exp2f(-s*phasediff*phasediff);
        return v * k;
    }

private:
    unsigned middle;
    unsigned sz;
};

class AddGaussOperator: public Gauss
{
public:
    Gauss g;
    __device__ void operator()(float& e, float2 const& v)
    {
        e += g.gauss_value(v);
    }
};


void addGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    AddGaussOperator gauss;
    gauss.g = g;

    element_operate<float, AddGaussOperator>(image, imageArea, gauss);
}


class MultiplyGaussOperator: public Gauss
{
public:
    Gauss g;
    __device__ void operator()(float& e, float2 const& v)
    {
        e *= exp2f(g.gauss_value(v));
    }
};


void multiplyGauss( float4 imageArea, cudaPitchedPtrType<float> image, Gauss g )
{
    MultiplyGaussOperator gauss;
    gauss.g = g;

    element_operate<float, MultiplyGaussOperator>(image, imageArea, gauss);
}
*/
