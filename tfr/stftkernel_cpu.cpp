#ifndef USE_CUDA
#include <cpumemorystorage.h>

#include "stftkernel.h"

void stftNormalizeInverse(
        DataStorage<float>::Ptr wavep,
        unsigned length )
{
    CpuMemoryReadWrite<float, 2> in_wt = CpuMemoryStorage::ReadWrite<2>( wavep );

    float v = 1.f/length;

#pragma omp parallel for
    for (int y=0; y<(int)in_wt.numberOfElements().height; ++y)
    {
        CpuMemoryReadWrite<float, 2>::Position pos( 0, y );
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

#pragma omp parallel for
    for (int y=0; y<(int)in_wt.numberOfElements().height; ++y)
    {
        CpuMemoryReadWrite<Tfr::ChunkElement, 2>::Position pos( 0, y );
        for (pos.x=0; pos.x<in_wt.numberOfElements().width; ++pos.x)
        {
            out_wt.write(pos, in_wt.ref(pos).real()*v);
        }
    }
}


void stftToComplex(
        DataStorage<float>::Ptr inwave,
        Tfr::ChunkData::Ptr outwave )
{
    CpuMemoryReadOnly<float, 2> in_wt = CpuMemoryStorage::ReadOnly<2>( inwave );
    CpuMemoryWriteOnly<Tfr::ChunkElement, 2> out_wt = CpuMemoryStorage::WriteAll<2>( outwave );

    int h = in_wt.numberOfElements().height,
        w = in_wt.numberOfElements().width;

    float *in = in_wt.ptr();
    Tfr::ChunkElement *out = out_wt.ptr();

#pragma omp parallel for
    for (int y=0; y<h; ++y)
    {
        for (int x=0; x<w; ++x)
            out[y*w+x] = Tfr::ChunkElement(in[y*w+x], 0.f);
    }
}


void cepstrumPrepareCepstra(
        Tfr::ChunkData::Ptr chunk,
        float normalization )
{
    CpuMemoryReadWrite<Tfr::ChunkElement, 2> cepstra = CpuMemoryStorage::ReadWrite<2>( chunk );

#pragma omp parallel for
    for (int y=0; y<(int)cepstra.numberOfElements().height; ++y)
    {
        CpuMemoryReadWrite<Tfr::ChunkElement, 2>::Position pos( 0, y );
        for (pos.x=0; pos.x<cepstra.numberOfElements().width; ++pos.x)
        {
            cepstra.write(pos, Tfr::ChunkElement(logf( 0.001f + norm(cepstra.ref(pos)) ) * normalization, 0));
        }
    }
}

#endif
