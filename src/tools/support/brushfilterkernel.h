#pragma once

#include "tfr/chunkdata.h"
#include "resampletypes.h"

void multiply( ResampleArea cwtArea, Tfr::ChunkData::Ptr cwt,
               ResampleArea imageArea, DataStorage<float>::Ptr image );
