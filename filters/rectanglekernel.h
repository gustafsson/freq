#ifndef RECTANGLE_CU_H
#define RECTANGLE_CU_H

#include "tfr/chunkdata.h"

struct Area
{
    float x1, y1, x2, y2;
};

void        removeRect( Tfr::ChunkData::Ptr wavelet, Area area, bool save_inside );

#endif // RECTANGLE_CU_H
