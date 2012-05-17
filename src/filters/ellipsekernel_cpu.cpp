#ifndef USE_CUDA
#include "ellipsekernel.h"
#include "ellipsekerneldef.h"

#include "cpumemorystorage.h"

#include <stdio.h>

void removeDisc( Tfr::ChunkData::Ptr waveletp, Area area, bool save_inside, float fs )
{
    Tfr::ChunkElement* wavelet = CpuMemoryStorage::ReadWrite<2>( waveletp ).ptr();

    DataStorageSize size = waveletp->size();

    DataPos p(0,0);
    for (p.y=0; p.y<size.height; ++p.y)
    {
        // size.width*2: To coalesce better, one thread for each float (instead of each float2)
        for (p.x=0; p.x<size.width*2; ++p.x)
        {
            remove_disc_elem(p, wavelet, size, area, save_inside, fs );
        }
    }
}
#endif // USE_CUDA
