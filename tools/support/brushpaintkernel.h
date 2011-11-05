#pragma once

#include "heightmap/blockkernel.h"

#include "resampletypes.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

class Gauss
{
public:
    Gauss(ResamplePos pos, ResamplePos sigma)
        :   pos(pos)
    {
        normalized_scale(sigma);
        k.x = M_LOG2E*0.5/(sigma.x*sigma.x);
        k.y = M_LOG2E*0.5/(sigma.y*sigma.y);
    }


    Gauss(ResamplePos pos, ResamplePos sigma, float scale)
        :   pos(pos),
            scale(scale)
    {
        k.x = M_LOG2E*0.5/(sigma.x*sigma.x);
        k.y = M_LOG2E*0.5/(sigma.y*sigma.y);
    }


    float gauss_value(float x, float y)
    {
        return gauss_value(ResamplePos(x,y));
    }


    RESAMPLE_ANYCALL float gauss_value(ResamplePos const& v)
    {
        ResamplePos r(v.x - pos.x, v.y - pos.y);
        // k = log2f(e)*0.5f/sigma/sigma
        r.x = r.x*r.x*k.x;
        r.y = r.y*r.y*k.y;
        return scale*exp2f(-r.x-r.y);
    }


    ResamplePos sigma()
    {
        // k = log2f(e)*0.5f/sigma/sigma
        return ResamplePos(
                sqrtf( M_LOG2E*0.5f/k.x ),
                sqrtf( M_LOG2E*0.5f/k.y )
            );
    }

    ResamplePos pos; // mu
    ResamplePos k; // k = log2f(e)*0.5f/sigma/sigma
    float scale;

private:
    void normalized_scale(ResamplePos sigma)
    {
        // this normalizes the bivariate gaussian
        scale = 1.0/(2.0*M_PI*sigma.x*sigma.y);
    }
};


void addGauss(
        ResampleArea imageArea, DataStorage<float>::Ptr image, Gauss gauss );
void multiplyGauss(
        ResampleArea imageArea, DataStorage<float>::Ptr image, Gauss gauss, Heightmap::AmplitudeAxis );
