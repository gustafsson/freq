#ifndef ELLIPSEFILTERDEF_H
#define ELLIPSEFILTERDEF_H

#include "ellipsekernel.h"
#include "resampletypes.h"
#include "datastorageaccess.h"

template<typename T>
inline RESAMPLE_ANYCALL void remove_disc_elem(DataPos p, T* wavelet, DataStorageSize numElem, Area area, bool save_inside, float fs )
{
    unsigned x = p.x, fi = p.y;

    bool complex = x%2;
    x/=2;

    if (x>=numElem.width )
        return;

    float rx = fabs(area.x2 - area.x1);
    float ry = fabs(area.y2 - area.y1);
    //float dx = fabs(x+.5f - area.x);
    //float dy = fabs(fi-.5f - area.y);
    float dx = fabs(x - area.x1);
    float dy = fabs(fi - area.y1);

    float ax = 0.03f*fs; // TODO this should be wavelet_time_support_samples( fs, hz ) = k*2^((b+fi)/scales_per_octave)
    float ay = 1.5f; // only round in time?

    // round corners
    float f = dx*dx/rx/rx + dy*dy/ry/ry;

    rx += ax;
    ry += ay;

    float g = dx*dx/rx/rx + dy*dy/ry/ry;
    if (f < 1) {
        f = 0;
    } else if (g<1) {
      f = (1 - 1/f) / (1/g - 1/f);
    } else {
      f = 1;
    }

    if (save_inside)
        f = 1-f;

    if (f < 1) {
        f = 3*f*f - 2*f*f*f;
        //f*=(1-f);
        //f*=(1-f);

        if (f != 0)
            f *= ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ];

        ((float*)wavelet)[ 2*x + complex + fi*2*numElem.width ] = f;
    }
}

#endif // ELLIPSEFILTERDEF_H
