#ifndef SPLINEFILTERKERNELDEF_H
#define SPLINEFILTERKERNELDEF_H

#include "splinefilterkernel.h"
#include <operate.h>

template<typename Reader, typename T>
class Spliner
{
public:
    Spliner(Reader reader, unsigned N, bool save_inside, float inv_fs)
        :   reader(reader),
            N(N),
            save_inside(save_inside),
            inv_fs(inv_fs)
    {}


    RESAMPLE_CALL void operator()(T& e, ResamplePos const& v)
    {
        // Count the number of times a line from v to infinity crosses the spline

        // Walk along +y axis only
        bool inside = false;
        float mindisty = FLT_MAX;
        float mindistx = FLT_MAX;
        for (unsigned i=0; i<N; ++i)
        {
            unsigned j = (i+1)%N;
            T pr = reader(i), qr = reader(j);
#ifdef __CUDACC__
            ResamplePos p(pr.x, pr.y), q(qr.x, qr.y);
#else
            ResamplePos p(pr.real(), pr.imag()), q(qr.real(), qr.imag());
#endif
            float r = (v.x - p.x)/(q.x - p.x);
            if (0.f <= r && 1.f > r)
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
            if (0.f <= r && 1.f > r)
            {
                float x = p.x + (q.x-p.x)*r;
                if (mindistx > fabsf(x-v.x))
                    mindistx = fabsf(x-v.x);
            }
        }

        float d = 1.f;
        if (inside)
        {
            d = 1 - mindisty*.25f - mindistx*inv_fs*20.f;

            if (d < 0.f)
                d = 0.f;
        }

        if (save_inside)
            d = 1.f - d;

        if (1.f != d)
            e *= d;
    }


private:
    Reader reader;
    unsigned N;
    bool save_inside;
    float inv_fs;
};


#endif // SPLINEFILTERKERNELDEF_H
