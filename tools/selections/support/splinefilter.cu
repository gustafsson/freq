#include "splinefilter.cu.h"
#include <resamplecuda.cu.h>
#include <operate.h>
#include "cudaglobalstorage.h"

template<typename Reader>
class Spliner
{
public:
    Spliner(Reader reader, unsigned N, bool save_inside)
        :   reader(reader),
            N(N),
            save_inside(save_inside)
    {}


    __device__ void operator()(float2& e, ResamplePos const& v)
    {
        // Count the number of times a line from v to infinity crosses the spline

        // Walk along +y axis only
        bool inside = false;
        float mindisty = FLT_MAX;
        float mindistx = FLT_MAX;
        for (unsigned i=0; i<N; ++i)
        {
            unsigned j = (i+1)%N;
            float2 p = reader(i), q = reader(j);
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

            float2 f = e;
            e = make_float2( f.x*d, f.y*d );
        }
    }


private:
    Reader reader;
    unsigned N;
    bool save_inside;
};


void applyspline(
        Tfr::ChunkData::Ptr data,
        DataStorage<Tfr::ChunkElement>::Ptr splinep, bool save_inside )
{
    cudaPitchedPtrType<float2> spline( CudaGlobalStorage::ReadOnly<1>(splinep).getCudaPitchedPtr());

    DataStorage<float2>::Ptr data2 =
            CudaGlobalStorage::BorrowPitchedPtr<float2>(
                    data->size(),
                    CudaGlobalStorage::ReadOnly<2>( data ).getCudaPitchedPtr()
                    );

    Spliner< Read1D<float2> > spliner(
            Read1D_Create<float2>( spline ),
            spline.getNumberOfElements().x,
            save_inside );

    element_operate<float2>( data2, ResampleArea(0, 0, data2->size().width, data2->size().height), spliner );

    Read1D_UnbindTexture<float2>();
}
