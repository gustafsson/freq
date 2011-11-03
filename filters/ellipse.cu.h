#ifndef ELLIPSE_CU_H
#define ELLIPSE_CU_H

#include "tfr/chunkdata.h"

struct Area
{
    float x1, y1, x2, y2;
};

void        removeDisc( Tfr::ChunkData::Ptr wavelet, Area area, bool _save_inside, float fs );

#endif // ELLIPSE_CU_H
