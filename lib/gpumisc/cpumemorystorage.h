#ifndef CPUMEMORYSTORAGE_H
#define CPUMEMORYSTORAGE_H

#include "datastorage.h"
#include "cpumemoryaccess.h"

class CpuMemoryStorage: public DataStorageImplementation
{
public:
    template<unsigned Dimension, typename T> static
    CpuMemoryReadOnly<T, Dimension> ReadOnly( boost::shared_ptr<DataStorage<T> >p )
    {
        return ReadOnly<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CpuMemoryReadOnly<T, Dimension> ReadOnly( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CpuMemoryStorage>( true, false )->Access<T, Dimension>();
    }


    template<unsigned Dimension, typename T> static
    CpuMemoryWriteOnly<T, Dimension> ReadWrite( boost::shared_ptr<DataStorage<T> > p )
    {
        return ReadWrite<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CpuMemoryReadWrite<T, Dimension> ReadWrite( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CpuMemoryStorage>( true, true )->Access<T, Dimension>();
    }


    template<unsigned Dimension, typename T> static
    CpuMemoryWriteOnly<T, Dimension> WriteAll( boost::shared_ptr<DataStorage<T> > p )
    {
        return WriteAll<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CpuMemoryWriteOnly<T, Dimension> WriteAll( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CpuMemoryStorage>( false, true )->Access<T, Dimension>();
    }


    template<typename T, unsigned Dimension>
    CpuMemoryAccess<T, Dimension> Access()
    {
        EXCEPTION_ASSERT( sizeof(T) == dataStorage()->bytesPerElement() );
        return CpuMemoryAccess<T, Dimension>((T*)data, size());
    }

    template<unsigned Dimension>
    CpuMemoryAccess<char, Dimension> AccessBytes()
    {
        return CpuMemoryAccess<char, Dimension>((char*)data, dataStorage()->sizeInBytes());
    }


    CpuMemoryStorage( DataStorageVoid* p, bool, bool );
    CpuMemoryStorage( DataStorageVoid* p, void* data, bool adoptData=false );
    ~CpuMemoryStorage(); // deleted through DataStorageImplementation


    template<typename T>
    static boost::shared_ptr<DataStorage<T> > BorrowPtr( DataStorageSize size, T* data, bool adoptData=false)
    {
        boost::shared_ptr<DataStorage<T> > ds( new DataStorage<T>(size) );
        new CpuMemoryStorage( ds.get(), data, adoptData); // Memory managed by DataStorage
        return ds;
    }

private:
    virtual bool updateFromOther(DataStorageImplementation *p);
    virtual bool updateOther(DataStorageImplementation *p);
    virtual void clear();
    virtual DataStorageImplementation* newInstance( DataStorageVoid* p );
    virtual bool allowCow();

    void* data;
    size_t data_N;

    bool borrowsData;
};



#endif // CPUMEMORYSTORAGE_H
