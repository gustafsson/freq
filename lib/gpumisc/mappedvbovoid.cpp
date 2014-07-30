#include "mappedvbovoid.h"

#include "gl.h" // cuda_gl_interop.h includes gl.h which expects windows.h to
                // be included on windows
#include "tasktimer.h"
#include "computationkernel.h"

#ifdef USE_CUDA
// cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudaglobalstorage.h"

#ifndef CUDA_MEMCHECK_TEST
//#define CUDA_MEMCHECK_TEST
#endif

#else

#include "cpumemorystorage.h"
#endif

// using TIME_MAPPEDVBO_LOG will break the flow in log files as the TaskTimer stays alive longer than it's scope
//#define TIME_MAPPEDVBO_LOG
#define TIME_MAPPEDVBO_LOG if(0)

//#define TIME_MAPPEDVBOVOID
#define TIME_MAPPEDVBOVOID if(0)

MappedVboVoid::
        MappedVboVoid(pVbo vbo)
            :
            _vbo(vbo),
            _is_mapped(false),
            _tt(0)
{
}


MappedVboVoid::
        ~MappedVboVoid()
{
}


void MappedVboVoid::
        map(DataStorageVoid* datap)
{
    TIME_MAPPEDVBO_LOG _tt = new TaskTimer("Mapping vbo %u of size %s", (unsigned)*_vbo, DataStorageVoid::getMemorySizeText(datap->numberOfBytes()).c_str());

    TIME_MAPPEDVBOVOID TaskTimer tt("Mapping vbo %u of size %s", (unsigned)*_vbo, DataStorageVoid::getMemorySizeText(datap->numberOfBytes()).c_str());

    EXCEPTION_ASSERT( !_is_mapped );

    DataStorageSize sizeInBytes = datap->sizeInBytes();
    EXCEPTION_ASSERT( datap->numberOfBytes() == _vbo->size() );

#ifdef USE_CUDA
    void* g_data=0;
    _vbo->registerWithCuda();
    _is_mapped = (cudaSuccess == cudaGLMapBufferObject((void**)&g_data, *_vbo));


    cudaPitchedPtr cpp;
    cpp.ptr = g_data;
    cpp.pitch = sizeInBytes.width;
    cpp.xsize = sizeInBytes.width;
    cpp.ysize = sizeInBytes.height*sizeInBytes.depth;


    if (!_is_mapped)
        mapped_gl_mem.reset( new DataStorage<char>( sizeInBytes ));
    else
        mapped_gl_mem = CudaGlobalStorage::BorrowPitchedPtr<char>( sizeInBytes, cpp );


    EXCEPTION_ASSERT( 0==datap->FindStorage<CudaGlobalStorage>() );

    #ifdef CUDA_MEMCHECK_TEST
        *datap = *mapped_gl_mem;
    #else
        if (_is_mapped)
            new CudaGlobalStorage( datap, cpp, false ); // Memory managed by DataStorage
    #endif
#else

    glBindBuffer(_vbo->vbo_type(), *_vbo);
#ifdef GL_ES_VERSION_2_0
    #ifndef GL_WRITE_ONLY
        #define GL_WRITE_ONLY 0x88B9 // from gl.h
    #endif
    void* data = glMapBufferOES(_vbo->vbo_type(), GL_WRITE_ONLY);
//    void* data = glMapBufferRange (_vbo->vbo_type(), 0, _vbo->size (),
//                                   GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
#else
    void* data = glMapBuffer(_vbo->vbo_type(), GL_WRITE_ONLY);
#endif
    _is_mapped = 0!=data;
    glBindBuffer(_vbo->vbo_type(), 0);


    if (!_is_mapped)
        mapped_gl_mem.reset( new DataStorage<char>( sizeInBytes ));
    else
        mapped_gl_mem = CpuMemoryStorage::BorrowPtr<char>( sizeInBytes, (char*)data );


    EXCEPTION_ASSERT( 0==datap->FindStorage<CpuMemoryStorage>() );

    if (_is_mapped)
        new CpuMemoryStorage( datap, data, false ); // Memory managed by DataStorage
#endif

    TIME_MAPPEDVBOVOID ComputationSynchronize();
}


void MappedVboVoid::
        unmap(DataStorageVoid* datap)
{
    if (_is_mapped)
    {
        TIME_MAPPEDVBOVOID TaskInfo("Unmapping vbo %u of size %s", (unsigned)*_vbo, DataStorageVoid::getMemorySizeText(datap->numberOfBytes()).c_str());

#ifdef USE_CUDA
        // make sure data is located in cuda
        datap->AccessStorage<CudaGlobalStorage>( true, false );

    #ifdef CUDA_MEMCHECK_TEST
        // copy data back over the mapped memory
        *mapped_gl_mem = *datap;
    #endif

        // sync from cuda to vbo
        cudaGLUnmapBufferObject(*_vbo);

        // release resources
        mapped_gl_mem.reset();
        datap->DiscardAllData();
        _is_mapped = false;

        // The memory bound with Cuda-OpenGL-interop can be relied on. So
        // call cudaGetLastError to clear the cuda error state just in case.
        // (I'm not sure why but it might be related to cuda out-of-memory
        // errors elsewhere)
        cudaGetLastError();
#else
        // make sure data is located in cpu
        datap->AccessStorage<CpuMemoryStorage>( true, false );

        // sync from mem to vbo
        glBindBuffer(_vbo->vbo_type(), *_vbo);
#ifdef GL_ES_VERSION_2_0
        glUnmapBufferOES(_vbo->vbo_type());
#else
        glUnmapBuffer(_vbo->vbo_type());
#endif
        glBindBuffer(_vbo->vbo_type(), 0);

        // release resources
        mapped_gl_mem.reset();
        datap->DiscardAllData();
        _is_mapped = false;
#endif

        TIME_MAPPEDVBOVOID ComputationSynchronize();

        if (_tt)
        {
            TaskInfo("Unmapped vbo %u of size %s", (unsigned)*_vbo, DataStorageVoid::getMemorySizeText(datap->numberOfBytes()).c_str());
            delete _tt;
            _tt = 0;
        }
    }
}
