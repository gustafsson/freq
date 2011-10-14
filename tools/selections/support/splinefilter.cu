#include "splinefilter.cu.h"
#include <operate.cu.h>
#include <resample.cu.h>

template<typename Reader>
class Spliner
{
public:
    Spliner(Reader reader, unsigned N, bool save_inside)
        :   reader(reader),
            N(N),
            save_inside(save_inside)
    {}


    __device__ void operator()(float2& e, float2 const& v)
    {
        // Count the number of times a line from v to infinity crosses the spline

        // Walk along +y axis only
        bool inside = false;
        float mindisty = FLT_MAX;
        float mindistx = FLT_MAX;
        for (unsigned i=0; i<N; ++i)
        {
            unsigned j = (i+1)%N;
            float2 p = reader(make_uint2(i,0)), q = reader(make_uint2(j,0));
            float r = (v.x - p.x)/(q.x - p.x);
            if (0 <= r && 1 > r)
            {
                float y = p.y + (q.y-p.y)*r;
                if (y > v.y)
                {
                    inside = !inside;
                }
                if (mindisty > fabsf(y-v.y))
                    mindisty = fabsf(y-v.y);
            }
            r = (v.y - p.y)/(q.y - p.y);
            if (0 <= r && 1 > r)
            {
                float x = p.x + (q.x-p.x)*r;
                if (mindistx > fabsf(x-v.x))
                    mindistx = fabsf(x-v.x);
            }
        }

        if (inside != save_inside)
        {
            float d = 1 - min(mindisty*(1/1.f), mindistx*(1/4.f));
            if (d < 0)
                d = 0;

            e.x *= d;
            e.y *= d;
        }
    }


private:
    Reader reader;
    unsigned N;
    bool save_inside;
};


void applyspline(
        cudaPitchedPtrType<float2> data,
        cudaPitchedPtrType<float2> spline, bool save_inside )
{
    Spliner< Read1D<float2> > spliner(
            Read1D_Create<float2>( spline ),
            spline.getNumberOfElements().x,
            save_inside );

    elemSize3_t sz = data.getNumberOfElements();
    element_operate<float2>( data, make_float4(0, 0, sz.x, sz.y), spliner );

    Read1D_UnbindTexture<float2>();
}
