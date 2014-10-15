#include "cpumemorystorage.h"
#include "timer.h"
#include "largememorypool.h"

#include <string.h> // memset, memcpy

#define LOG_SLOW_ALLOCATION false

CpuMemoryStorage::
        CpuMemoryStorage(DataStorageVoid* p, bool, bool)
    :
    DataStorageImplementation( p ),
    data( 0 ),
    borrowsData( false )
{
    CpuMemoryStorage* q = p->FindStorage<CpuMemoryStorage>();
    EXCEPTION_ASSERT( q == this );

    if (LOG_SLOW_ALLOCATION) {
        Timer t;
        data_N = p->numberOfBytes();
        data = lmp_malloc (data_N);
        double T = t.elapsed ();
        static size_t C = 0;
        C++;
        if (1e-4 < T)
            Log("cpu: allocated #%d of %s in %s") % C % DataStorageVoid::getMemorySizeText (data_N) % TaskTimer::timeToString (T);
    } else {
        data_N = p->numberOfBytes();
        data = lmp_malloc (data_N);
    }
}


CpuMemoryStorage::
        CpuMemoryStorage( DataStorageVoid* p, void* data, bool adoptData )
    :
    DataStorageImplementation( p ),
    data( data ),
    data_N( 0 ), // not allocated through LargeMemoryPool
    borrowsData( !adoptData )
{
    CpuMemoryStorage* q = p->AccessStorage<CpuMemoryStorage>( false, true ); // Mark borrowed memory as up to date
    EXCEPTION_ASSERT( q == this );
}


CpuMemoryStorage::
        ~CpuMemoryStorage()
{
    if (!borrowsData) {
        if (LOG_SLOW_ALLOCATION) {
            Timer t;
            lmp_free (data, data_N);
            double T = t.elapsed ();
            static size_t C = 0;
            C++;
            if (1e-4 < T)
                Log("cpu: released #%d of %s in %s") % C % DataStorageVoid::getMemorySizeText (data_N) % TaskTimer::timeToString (T);
        } else {
            lmp_free (data, data_N);
        }
    }
}


bool CpuMemoryStorage::
        updateFromOther(DataStorageImplementation *p)
{
    // CpuMemoryStorage is the fundamental type and can't convert to/from any
    // other storage type
    CpuMemoryStorage* q = dynamic_cast<CpuMemoryStorage*>(p);
    if (!q)
        return false;

    EXCEPTION_ASSERT( q->dataStorage()->numberOfBytes() == dataStorage()->numberOfBytes() );

    if (data != q->data)
        memcpy( data, q->data, dataStorage()->numberOfBytes() );

    return true;
}


bool CpuMemoryStorage::
        updateOther(DataStorageImplementation *p)
{
    if (CpuMemoryStorage* c = dynamic_cast<CpuMemoryStorage*>(p))
        return c->updateFromOther( this );
    return false;
}


void CpuMemoryStorage::
        clear()
{
    memset( data, 0, dataStorage()->numberOfBytes() );
}


DataStorageImplementation* CpuMemoryStorage::
        newInstance( DataStorageVoid* p )
{
    return new CpuMemoryStorage( p, bool(), bool() );
}


bool CpuMemoryStorage::
        allowCow()
{
    return !borrowsData;
}
