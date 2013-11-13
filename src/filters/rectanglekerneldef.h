#ifndef RECTANGLEKERNELDEF_H
#define RECTANGLEKERNELDEF_H

#include "resample.h"
#include "rectanglekernel.h"

template<typename T>
inline RESAMPLE_CALL void remove_rect_elem(DataPos p, T* wavelet, DataStorageSize numElem, Area area, float save_inside )
{
    const int
            x = p.x,
            fi = p.y;

    if (x>=numElem.width )
        return;

    float f;

    if(x >= area.x1 && x <= area.x2 && fi >= area.y1 && fi <= area.y2)
    {
        f = save_inside;
    }
    else
    {
        f = !save_inside;
    }

    if (f == 0.f)
    {
        float dx = min(fabsf(x-area.x1), fabsf(x-area.x2));
        float dy = min(fabsf(fi-area.y1), fabsf(fi-area.y1));
        float f = 1.f - min(dy*(1/1.f), dx*(1/4.f));
        if (f < 0.f)
            f = 0.f;
    }

    wavelet[ x + fi*numElem.width ] *= f;
}


#endif // RECTANGLEKERNELDEF_H
