#pragma once

#include "tfr/chunkdata.h"

void applyspline(
        Tfr::ChunkData::Ptr data,
        DataStorage<Tfr::ChunkElement>::Ptr spline, bool save_inside );
