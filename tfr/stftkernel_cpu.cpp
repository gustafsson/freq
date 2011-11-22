#ifndef USE_CUDA
#include <cpumemorystorage.h>

#include "stftkernel.h"

void stftNormalizeInverse(
        DataStorage<float>::Ptr wavep,
        unsigned length )
{
    CpuMemoryReadWrite<float, 2> in_wt = CpuMemoryStorage::ReadWrite<2>( wavep );

    float v = 1.f/length;

    CpuMemoryReadWrite<float, 2>::Position pos( 0, 0 );
    for (pos.y=0; pos.y<in_wt.numberOfElements().height; ++pos.y)
    {
        for (pos.x=0; pos.x<in_wt.numberOfElements().width; ++pos.x)
        {
            in_wt.ref(pos) *= v;
        }
    }
}

void stftNormalizeInverse(
        Tfr::ChunkData::Ptr inwave,
        DataStorage<float>::Ptr outwave,
        unsigned length )
{
    CpuMemoryReadOnly<Tfr::ChunkElement, 2> in_wt = CpuMemoryStorage::ReadOnly<2>( inwave );
    CpuMemoryWriteOnly<float, 2> out_wt = CpuMemoryStorage::WriteAll<2>( outwave );

    float v = 1.f/length;

    CpuMemoryReadWrite<float, 2>::Position pos( 0, 0 );
    for (pos.y=0; pos.y<in_wt.numberOfElements().height; ++pos.y)
    {
        for (pos.x=0; pos.x<in_wt.numberOfElements().width; ++pos.x)
        {
            out_wt.write(pos, in_wt.ref(pos).real()*v);
        }
    }
}
#endif
