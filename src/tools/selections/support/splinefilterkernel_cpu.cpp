#ifndef USE_CUDA

#include "resamplecpu.h"

#include "splinefilterkerneldef.h"

void applyspline(
        Tfr::ChunkData::ptr data,
        DataStorage<Tfr::ChunkElement>::ptr splinep, bool save_inside, float fs )
{
    Spliner< CpuReader<Tfr::ChunkElement>, Tfr::ChunkElement > spliner(
            CpuReader<Tfr::ChunkElement>( CpuMemoryStorage::ReadOnly<2>(splinep) ),
            splinep->size().width,
            save_inside, 1/fs );

    element_operate<Tfr::ChunkElement>( data, ResampleArea(0, 0, data->size().width, data->size().height), spliner );
}

#endif // USE_CUDA
