#pragma once

#include "tfr/chunkdata.h"
#include "resampletypes.h"

void multiply( ResampleArea cwtArea, Tfr::ChunkData::ptr cwt,
               ResampleArea imageArea, DataStorage<float>::ptr image );
