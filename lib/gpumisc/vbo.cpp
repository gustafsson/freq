#include "gl.h"
#include "vbo.h"
#include "TaskTimer.h"
#include "datastorage.h" // getMemorySizeText

#ifdef USE_CUDA
#include "CudaException.h"
#include <cuda_gl_interop.h>
#endif
#include "GlException.h"


//#define TIME_VBO
#define TIME_VBO if(0)


Vbo::Vbo(size_t size, unsigned vbo_type, unsigned access_pattern, void* data)
:   _sz(0),
    _vbo(0),
#ifdef USE_CUDA
    _registered(false),
#endif
    _vbo_type(0)
{
    init(size, vbo_type, access_pattern, data);
}


Vbo::~Vbo()
{
    clear();
}

Vbo::
        operator GLuint() const
{
    return _vbo;
}


#ifdef USE_CUDA
void Vbo::
        registerWithCuda()
{
    if (!_registered)
    {
        TIME_VBO TaskTimer tt("Vbo::registerWithCuda(), %u, size %s", _vbo, DataStorageVoid::getMemorySizeText(_sz).c_str());
        CudaException_SAFE_CALL( cudaGLRegisterBufferObject(_vbo) );
    }
    else
    {
//        cudaGLRegisterBufferObject(_vbo);
//        cudaGetLastError();
    }

    _registered = true;
}
#endif


void Vbo::
        init(size_t size, unsigned vbo_type, unsigned access_pattern, void* data)
{
    TIME_VBO TaskTimer tt("Vbo::init(%s, %u, %u, %p)",
                          DataStorageVoid::getMemorySizeText(_sz).c_str(), vbo_type, access_pattern, data);

    GlException_CHECK_ERROR();

    clear();

    GlException_CHECK_ERROR();

    // create buffer object
    glGenBuffers(1, &_vbo);
    glBindBuffer(vbo_type, _vbo);
    glBufferData(vbo_type, size, data, access_pattern);
    glBindBuffer(vbo_type, 0);

    TIME_VBO TaskInfo("Got vbo %u", _vbo) ;

    GlException_CHECK_ERROR();

    _sz = size;
    _vbo_type = vbo_type;
}


void Vbo::
        clear()
{
    if (_vbo)
    {
        TIME_VBO TaskTimer tt("Vbo::clear %u, size %s", _vbo,
                              DataStorageVoid::getMemorySizeText(_sz).c_str());

#ifdef USE_CUDA
        cudaError_t e = cudaSuccess;
        if (_registered)
        {
            e = cudaGLUnregisterBufferObject(_vbo);
            _registered = false;
        }
#endif
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
        _sz = 0;
    }
}
