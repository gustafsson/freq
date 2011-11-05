#include <resamplecpu.h>

#include "splinefilterkerneldef.h"

void applyspline(
        Tfr::ChunkData::Ptr data,
        DataStorage<Tfr::ChunkElement>::Ptr splinep, bool save_inside )
{
    Spliner< CpuReader<Tfr::ChunkElement>, Tfr::ChunkElement > spliner(
            CpuReader<Tfr::ChunkElement>( CpuMemoryStorage::ReadOnly<2>(splinep) ),
            splinep->size().width,
            save_inside );

    element_operate<Tfr::ChunkElement>( data, ResampleArea(0, 0, data->size().width, data->size().height), spliner );
}

