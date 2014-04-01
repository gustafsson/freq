#pragma once

#include "tfr/chunkdata.h"

void applyspline(
        Tfr::ChunkData::ptr data,
        DataStorage<Tfr::ChunkElement>::ptr spline, bool save_inside, float fs );
