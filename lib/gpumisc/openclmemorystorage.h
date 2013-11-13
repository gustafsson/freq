#ifndef OPENCLMEMORYSTORAGE_H
#define OPENCLMEMORYSTORAGE_H

#include "datastorage.h"
#include "openclmemoryaccess.h"

#include <stdexcept>

class OpenClMemoryStorage : public DataStorageImplementation
{
public:
    template<unsigned Dimension, typename T> static
    OpenClMemoryAccess<T, Dimension> ReadOnly( boost::shared_ptr<DataStorage<T> >p )
    {
        return Access<T, Dimension>( p.get(), true, false );
    }


    template<unsigned Dimension, typename T> static
    OpenClMemoryAccess<T, Dimension> ReadWrite( boost::shared_ptr<DataStorage<T> > p )
    {
        return Access<T, Dimension>( p.get(), true, true );
    }


    template<unsigned Dimension, typename T> static
    OpenClMemoryAccess<T, Dimension> WriteAll( boost::shared_ptr<DataStorage<T> > p )
    {
        return Access<T, Dimension>( p.get(), false, true );
    }


    template<typename T, unsigned Dimension> static
    OpenClMemoryAccess<T, Dimension> Access( DataStorageVoid* dsv, bool read, bool write )
    {
        return AccessStorage(dsv, read, write)->Access<T, Dimension>();
    }


    // TODO rename to AccessClStorage
    // validates read/write versus cl_mem_flags
    static OpenClMemoryStorage* AccessStorage( DataStorageVoid* dsv, bool read, bool write );


    template<typename T, unsigned Dimension>
    OpenClMemoryAccess<T, Dimension> Access()
    {
        EXCEPTION_ASSERT( sizeof(T) == dataStorage()->bytesPerElement() );
        return OpenClMemoryAccess<T, Dimension>(data, size());
    }


    OpenClMemoryStorage( DataStorageVoid* p, bool read, bool write );
    OpenClMemoryStorage( DataStorageVoid* p, cl_mem data, cl_mem_flags flags, bool adoptData=false );
    ~OpenClMemoryStorage(); // deleted through DataStorageImplementation


    template<typename T>
    static boost::shared_ptr<DataStorage<T> > BorrowPtr( DataStorageSize size, cl_mem data, cl_mem_flags flags, bool adoptData=false)
    {
        boost::shared_ptr<DataStorage<T> > ds( new DataStorage<T>(size) );
        new OpenClMemoryStorage( ds.get(), data, flags, adoptData); // Memory managed by DataStorage
        return ds;
    }

private:
    virtual bool updateFromOther(DataStorageImplementation *p);
    virtual bool updateOther(DataStorageImplementation *p);
    virtual void clear();
    virtual DataStorageImplementation* newInstance( DataStorageVoid* p );
    virtual bool allowCow();

    cl_mem data;
    cl_mem_flags flags;

    bool borrowsData;
};

#endif // OPENCLMEMORYSTORAGE_H
