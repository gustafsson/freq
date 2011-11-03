#pragma once

#include <cudaPitchedPtrType.h>

#include "tfr/chunkdata.h"

struct ImageArea
{
    float t1, s1, t2, s2;
};

void multiply( ImageArea cwtArea, Tfr::ChunkData::Ptr cwt,
               ImageArea imageArea, DataStorage<float>::Ptr image );
