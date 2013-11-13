#include "cudaglobalstorage.h"

#include "CudaException.h"
#include <cuda_runtime.h>
#include "cudaMemcpy3Dfix.h"

#include "cpumemorystorage.h"

#include <string.h> // memset


// ugly but worth it
storageCudaMemsetFixT storageCudaMemsetFix;


CudaGlobalStorage::
        CudaGlobalStorage( DataStorageVoid* p, bool, bool, bool allocateWithPitch )
            :
            DataStorageImplementation( p ),
            borrowsData( false ),
            allocatedWithPitch( allocateWithPitch )
{
    CudaGlobalStorage* q = p->FindStorage<CudaGlobalStorage>();
    EXCEPTION_ASSERT( q == this );

    memset( &data, 0, sizeof(data) );
    DataStorageSize sz = size();
    cudaExtent x = make_cudaExtent( sz.width*dataStorage()->bytesPerElement(), sz.height, sz.depth );

    CudaException_CHECK_ERROR();
    if (allocateWithPitch)
        CudaException_SAFE_CALL( cudaMalloc3D( &data, x ) );
    else
    {
        CudaException_SAFE_CALL( cudaMalloc( &data.ptr, x.width*x.height*x.depth ) );
        data.pitch = x.width;
        data.xsize = x.width;
        data.ysize = x.height*x.depth;
    }

    getCudaExtent(); // performs some checks
}


CudaGlobalStorage::
        CudaGlobalStorage( DataStorageVoid* p, cudaPitchedPtr externalData, bool adoptData )
            :
            DataStorageImplementation( p ),
            borrowsData( !adoptData ),
            allocatedWithPitch( externalData.pitch != externalData.xsize )
{
    CudaGlobalStorage* q = p->FindStorage<CudaGlobalStorage>();
    EXCEPTION_ASSERT( q == this );

    DataStorageSize sz = size();
    EXCEPTION_ASSERT( (int)externalData.xsize == sz.width*dataStorage()->bytesPerElement() );
    EXCEPTION_ASSERT( (int)externalData.ysize == sz.depth*sz.height );
    EXCEPTION_ASSERT( externalData.ptr != 0 );

    data = externalData;

    p->FindCreateStorage<CudaGlobalStorage>( false, true ); // Mark memory as up to date
}


CudaGlobalStorage::
        ~CudaGlobalStorage()
{
    if (!borrowsData)
        CudaException_SAFE_CALL( cudaFreeAndZero( &data.ptr ));
}


cudaExtent CudaGlobalStorage::
        getCudaExtent()
{
    DataStorageSize sz = size();
    EXCEPTION_ASSERT( (int)data.xsize == sz.width*dataStorage()->bytesPerElement() );
    EXCEPTION_ASSERT( (int)data.ysize == sz.height*sz.depth );
    EXCEPTION_ASSERT( data.xsize <= data.pitch );
    return make_cudaExtent( data.xsize, sz.height, sz.depth );
}


/*static*/ void CudaGlobalStorage::
        useCudaPitch( DataStorageVoid* dsv, bool allocateWithPitch )
{
    CudaGlobalStorage* cgs = dsv->FindStorage<CudaGlobalStorage>();
    if (cgs)
    {
        if (cgs->allocatedWithPitch == allocateWithPitch)
            return;

        if (!allocateWithPitch && cgs->data.pitch == cgs->data.xsize)
        {
            cgs->allocatedWithPitch = false;
            return;
        }

        delete cgs;
    }

    new CudaGlobalStorage( dsv, bool(), bool(), allocateWithPitch ); // Memory managed by DataStorage
}


bool CudaGlobalStorage::
        updateFromOther(DataStorageImplementation *p)
{
    if (CpuMemoryStorage* cpu = dynamic_cast<CpuMemoryStorage*>(p))
    {
        CpuMemoryAccess<char, 3> cpuReader = cpu->AccessBytes<3>();

        cudaMemcpy3DParms params;
        memset( &params, 0, sizeof(cudaMemcpy3DParms));

        params.srcPtr = make_cudaPitchedPtr(
                cpuReader.ptr(),
                cpuReader.numberOfElements().width, // elements are bytes
                cpuReader.numberOfElements().width, // elements are bytes
                cpuReader.numberOfElements().height);

        params.dstPtr = data;
        params.extent = getCudaExtent();
        params.kind = cudaMemcpyHostToDevice;

        CudaException_SAFE_CALL( cudaMemcpy3Dfix( &params ) );
        return true;
    }

    if (CudaGlobalStorage* b = dynamic_cast<CudaGlobalStorage*>(p))
    {
        cudaMemcpy3DParms params;
        memset( &params, 0, sizeof(cudaMemcpy3DParms));

        params.srcPtr = b->data;
        params.dstPtr = data;
        params.extent = getCudaExtent();
        params.kind = cudaMemcpyDeviceToDevice;

        CudaException_SAFE_CALL( cudaMemcpy3Dfix( &params ) );
        return true;
    }

    return false;
}


bool CudaGlobalStorage::
        updateOther(DataStorageImplementation *p)
{
    if (CpuMemoryStorage* cpu = dynamic_cast<CpuMemoryStorage*>(p))
    {
        CpuMemoryAccess<char, 3> cpuWriter = cpu->AccessBytes<3>();

        cudaMemcpy3DParms params;
        memset( &params, 0, sizeof(cudaMemcpy3DParms));

        params.srcPtr = data;

        params.dstPtr = make_cudaPitchedPtr(
                cpuWriter.ptr(),
                cpuWriter.numberOfElements().width,
                cpuWriter.numberOfElements().width,
                cpuWriter.numberOfElements().height);

        params.extent = getCudaExtent();
        params.kind = cudaMemcpyDeviceToHost;

        CudaException_SAFE_CALL( cudaMemcpy3Dfix( &params ) );

        return true;
    }

    if (CudaGlobalStorage* b = dynamic_cast<CudaGlobalStorage*>(p))
    {
        return b->updateFromOther( this );
    }

    return false;
}


void CudaGlobalStorage::
        clear()
{
    if (storageCudaMemsetFix)
        storageCudaMemsetFix(data.ptr, data.pitch*data.ysize*size().depth);
    else
        CudaException_SAFE_CALL( cudaMemset3D( data, 0, getCudaExtent() ) );
}


DataStorageImplementation* CudaGlobalStorage::
        newInstance( DataStorageVoid* p )
{
    return new CudaGlobalStorage( p, bool(), bool(), allocatedWithPitch );
}


bool CudaGlobalStorage::
        allowCow()
{
    return !borrowsData;
}
