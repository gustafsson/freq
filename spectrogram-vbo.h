#ifndef SPECTROGRAMVBO_H
#define SPECTROGRAMVBO_H

#include <cuda_runtime.h>
#ifdef _MSC_VER
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include <boost/shared_ptr.hpp>
#include <GpuCpuData.h>
#include <stdio.h>
#include <TaskTimer.h>

typedef boost::shared_ptr<class SpectrogramVbo> pSpectrogramVbo;
typedef boost::shared_ptr<class SpectrogramRenderer> pSpectrogramRenderer;
typedef boost::shared_ptr<class Spectrogram> pSpectrogram;

typedef boost::shared_ptr< class Vbo > pVbo;

GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

class Vbo
{
public:
    Vbo(size_t sz);
    ~Vbo();
    operator GLuint() const { return _vbo; }

    size_t size() { return _sz; }
private:
    Vbo(const Vbo &b);
    Vbo& operator=(const Vbo &b);

    size_t _sz;
    GLuint _vbo;
};


template<typename T>
class MappedVbo
{
public:
    MappedVbo( pVbo vbo, cudaExtent numberOfElements )
    :   _vbo(vbo),
        tt("Mapping vbo")
    {
        void* g_data;
        cudaGLMapBufferObject((void**)&g_data, *_vbo);

        cudaPitchedPtr cpp;
        cpp.ptr = g_data;
        cpp.pitch = sizeof(T)*numberOfElements.width;
        cpp.xsize = sizeof(T)*numberOfElements.width;
        cpp.ysize = numberOfElements.height;
        data.reset( new GpuCpuData<T>( &cpp, numberOfElements, GpuCpuVoidData::CudaGlobal, true ));
        BOOST_ASSERT( data->getSizeInBytes1D() == vbo->size() );
    }

    ~MappedVbo() {
        data->getCudaGlobal();
        cudaGLUnmapBufferObject(*_vbo);
    }

    boost::shared_ptr<GpuCpuData<T> > data;

private:
    pVbo _vbo;

    TaskTimer tt;
};

class SpectrogramVbo
{
public:
    SpectrogramVbo( Spectrogram* spectrogram );
    ~SpectrogramVbo();

    typedef boost::shared_ptr< MappedVbo<float> > pHeight;
    typedef boost::shared_ptr< MappedVbo<float2> > pSlope;

    pHeight height();
    pSlope slope();
    void unmap();

    void draw( );
    void draw_directMode( );
private:
    Spectrogram* _spectrogram;

    pHeight _mapped_height;
    pSlope _mapped_slope;

    pVbo _height;
    pVbo _slope;
};


#endif // SPECTROGRAMVBO_H
