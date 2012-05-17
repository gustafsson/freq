#ifndef USE_CUDA
#include "resamplecpu.h"
#include "rectanglekerneldef.h"

void removeRect( Tfr::ChunkData::Ptr waveletp, Area area, bool save_inside )
{
    Tfr::ChunkElement* wavelet = CpuMemoryStorage::ReadWrite<2>( waveletp ).ptr();
    DataStorageSize size = waveletp->size();

    DataPos p(0,0);
    for (p.y=0; p.y<size.height; ++p.y)
    {
        for (p.x=0; p.x<size.width; ++p.x)
        {
            remove_rect_elem(p, wavelet, size, area, save_inside );
        }
    }
}

#endif // USE_CUDA
