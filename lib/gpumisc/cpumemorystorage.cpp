#include "cpumemorystorage.h"

#include <string.h> // memset, memcpy

CpuMemoryStorage::
        CpuMemoryStorage(DataStorageVoid* p, bool, bool)
    :
    DataStorageImplementation( p ),
    data( 0 ),
    borrowsData( false )
{
    CpuMemoryStorage* q = p->FindStorage<CpuMemoryStorage>();
    EXCEPTION_ASSERT( q == this );

    data = new char[ p->numberOfBytes() ];
}


CpuMemoryStorage::
        CpuMemoryStorage( DataStorageVoid* p, void* data, bool adoptData )
    :
    DataStorageImplementation( p ),
    data( data ),
    borrowsData( !adoptData )
{
    CpuMemoryStorage* q = p->AccessStorage<CpuMemoryStorage>( false, true ); // Mark borrowed memory as up to date
    EXCEPTION_ASSERT( q == this );
}


CpuMemoryStorage::
        ~CpuMemoryStorage()
{
    if (!borrowsData)
        delete [](char*)data;
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
