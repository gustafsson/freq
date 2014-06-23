#ifndef CUDAGLOBALSTORAGE_H
#define CUDAGLOBALSTORAGE_H

#include "cudaglobalaccess.h"
#include "datastorage.h"

// ugly but worth it
typedef void (*storageCudaMemsetFixT)(void* p, int N);
extern storageCudaMemsetFixT storageCudaMemsetFix;

class CudaGlobalStorage: public DataStorageImplementation
{
public:
    template<unsigned Dimension, typename T> static
    CudaGlobalReadOnly<T, Dimension> ReadOnly( boost::shared_ptr< DataStorage<T> >p )
    {
        return ReadOnly<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CudaGlobalReadOnly<T, Dimension> ReadOnly( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CudaGlobalStorage>( true, false )->Access<T, Dimension>();
    }


    template<unsigned Dimension, typename T> static
    CudaGlobalReadWrite<T, Dimension> ReadWrite( boost::shared_ptr<DataStorage<T> > p )
    {
        return ReadWrite<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CudaGlobalReadWrite<T, Dimension> ReadWrite( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CudaGlobalStorage>( true, true )->Access<T, Dimension>();
    }


    template<unsigned Dimension, typename T> static
    CudaGlobalWriteOnly<T, Dimension> WriteAll( boost::shared_ptr<DataStorage<T> > p )
    {
        return WriteAll<T, Dimension>( p.get() );
    }

    template<typename T, unsigned Dimension> static
    CudaGlobalWriteOnly<T, Dimension> WriteAll( DataStorageVoid* dsv )
    {
        return dsv->AccessStorage<CudaGlobalStorage>( false, true )->Access<T, Dimension>();
    }


    template<typename T, unsigned Dimension>
    CudaGlobalAccess<T, Dimension> Access()
    {
        EXCEPTION_ASSERT( sizeof(T) == dataStorage()->bytesPerElement() );
        return CudaGlobalAccess<T, Dimension>(data, size().depth);
    }

    template<unsigned Dimension>
    CudaGlobalAccess<char, Dimension> AccessBytes()
    {
        return CudaGlobalAccess<char, Dimension>(data, size().depth);
    }

    CudaGlobalStorage( DataStorageVoid* p, bool, bool, bool allocateWithPitch=true  );
    CudaGlobalStorage( DataStorageVoid* p, cudaPitchedPtr data, bool adoptData=false );
    ~CudaGlobalStorage();


    template<typename T>
    static boost::shared_ptr<DataStorage<T> > BorrowPitchedPtr(cudaExtent size, cudaPitchedPtr data, bool adoptData=false)
    {
        return BorrowPitchedPtr<T>( DataStorageSize( size.width, size.height, size.depth ), data, adoptData );
    }

    template<typename T>
    static boost::shared_ptr<DataStorage<T> > BorrowPitchedPtr(uint3 size, cudaPitchedPtr data, bool adoptData=false)
    {
        return BorrowPitchedPtr<T>( DataStorageSize(size.x, size.y, size.z ), data, adoptData );
    }

    template<typename T>
    static boost::shared_ptr<DataStorage<T> > BorrowPitchedPtr(DataStorageSize size, cudaPitchedPtr data, bool adoptData=false )
    {
        boost::shared_ptr<DataStorage<T> > ds( new DataStorage<T>(size) );
        new CudaGlobalStorage( ds.get(), data, adoptData ); // Memory managed by DataStorage
        return ds;
    }


    cudaExtent getCudaExtent();


    template<typename T>
    static void useCudaPitch( boost::shared_ptr<DataStorage<T> > ds, bool allocateWithPitch = false )
    {
        useCudaPitch( ds.get(), allocateWithPitch );
    }

    static void useCudaPitch( DataStorageVoid* dsv, bool allocateWithPitch = false );

    virtual void clear();
private:
    virtual bool updateFromOther(DataStorageImplementation *p);
    virtual bool updateOther(DataStorageImplementation *p);
    virtual DataStorageImplementation* newInstance( DataStorageVoid* p );
    virtual bool allowCow();


    cudaPitchedPtr data;
    bool borrowsData;
    bool allocatedWithPitch;
};


#endif // CUDAGLOBALSTORAGE_H
