#ifndef STFTKERNEL_H
#define STFTKERNEL_H

#include "tfr/chunkdata.h"

void        stftNormalizeInverse( DataStorage<float>::Ptr wave, unsigned length );
void        stftNormalizeInverse( Tfr::ChunkData::Ptr inwave, DataStorage<float>::Ptr outwave, unsigned length );

#endif // STFTKERNEL_H
