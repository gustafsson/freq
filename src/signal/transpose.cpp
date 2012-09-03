#include "transpose.h"
#include "cpumemorystorage.h"

#include <string.h> // memcpy

namespace Signal {

void transpose(DataStorage<float>* dest, DataStorage<float>* src)
{
    DataStorageSize sz = dest->size();

    BOOST_ASSERT( dest->numberOfElements() == src->numberOfElements() );
    BOOST_ASSERT( sz.depth == 1 );

    float* destp = CpuMemoryStorage::WriteAll<float,3>( dest ).ptr();
    float* srcp = CpuMemoryStorage::ReadOnly<float,3>( src ).ptr();

    switch(sz.height)
    {
    case 1:
        memcpy(destp, srcp, sizeof(float)*sz.width);
        break;
    case 2:
        for (int i=0; i<sz.width; i++) {
            float *t = destp + i;
            float *d = srcp + i*2;
            t[0] = d[0];
            t[sz.width] = d[1];
        }
        break;
    default:
        switch(sz.width)
        {
        case 1:
            memcpy(destp, srcp, sizeof(float)*sz.height);
            break;
        case 2:
            for (int i=0; i<sz.height; i++) {
                float *t = destp + i*2;
                float *d = srcp + i;
                t[0] = d[0];
                t[1] = d[sz.height];
            }
            break;
        default:
            if (sz.height < sz.width)
            {
                for (int j=0; j<sz.height; j++) {
                    float *t = destp + j*sz.width;
                    float *d = srcp + j;
                    for (int i=0; i<sz.width; i++) {
                        t[i] = d[i*sz.height];
                    }
                }
            } else {
                for (int j=0; j<sz.width; j++) {
                    float *t = destp + j;
                    float *d = srcp + j*sz.height;
                    for (int i=0; i<sz.height; i++) {
                        t[i*sz.width] = d[i];
                    }
                }
            }
            break;
        }
        break;
    }
}

} // namespace Signal
