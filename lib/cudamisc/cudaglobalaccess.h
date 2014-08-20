#ifndef CUDAGLOBALACCESS_H
#define CUDAGLOBALACCESS_H

#ifdef __CUDACC__
#define ACCESSCALL __device__ __host__
#else
#define ACCESSCALL
#endif

#include "datastorageaccess.h"

#include <cuda_runtime.h>
#include "cudaPitchedPtrType.h"



template<typename DataType, unsigned Dimension>
class CudaGlobalAccess
{
    // implementes DataStorageAccessConcept

    DataType* ptr;
    DataAccessSize<Dimension> sz; // numberOfElements
    size_t pitch;

public:
    typedef DataType T;
    typedef DataAccessPosition<Dimension> Position;
    typedef DataAccessSize<Dimension> Size;


    CudaGlobalAccess(cudaPitchedPtr p, DataAccessPosition_t depth=1)
        :
        ptr((DataType*)p.ptr),
        sz(1),
        pitch(p.pitch)
    {
        sz = DataAccessSize<3>( p.xsize/sizeof(DataType),
                                     p.ysize/depth,
                                     depth );
    }


    ACCESSCALL DataAccessSize<Dimension> numberOfElements() const { return sz; }

    cudaPitchedPtr getCudaPitchedPtr() const
    {
        DataAccessSize<2> sz2 = sz;
        sz2.width *= sizeof(DataType);
        switch(Dimension)
        {
        case 1:
            return make_cudaPitchedPtr( ptr, pitch, sz2.width, 1 );
        case 2:
            return make_cudaPitchedPtr( ptr, pitch, sz2.width, sz2.height );
        case 3:
            return make_cudaPitchedPtr( ptr, pitch, sz2.width, sz2.height );
        }
        return make_cudaPitchedPtr( 0, 0, 0, 0 );
    }


    ACCESSCALL       T* device_ptr()       { return ptr; }
    ACCESSCALL const T* device_ptr() const { return ptr; }


    ACCESSCALL T& ref( Position p )
    {
        return ptr[ sz.offset(p) ];
    }
};


template<typename DataType, unsigned Dimension>
class CudaGlobalReadOnly: public CudaGlobalAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CudaGlobalAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CudaGlobalReadOnly(const Access& r)
        : Access(r)
    {}

    ACCESSCALL DataType read( const Position& p )
    {
        return ref( p );
    }
};


template<typename DataType, unsigned Dimension>
class CudaGlobalWriteOnly: public CudaGlobalAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CudaGlobalAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CudaGlobalWriteOnly(const Access& a)
        : Access(a)
    {}

    ACCESSCALL void write( const Position& p, const DataType& v )
    {
        ref(p) = v;
    }
};


template<typename DataType, unsigned Dimension>
class CudaGlobalReadWrite: public CudaGlobalAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CudaGlobalAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CudaGlobalReadWrite(const Access& a)
        : Access(a)
    {}

    ACCESSCALL DataType read( const Position& p )
    {
        return ref( p );
    }

    ACCESSCALL void write( const Position& p, const DataType& v )
    {
        ref(p) = v;
    }
};

#endif // CUDAGLOBALACCESS_H
