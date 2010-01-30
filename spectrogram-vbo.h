#ifndef SPECTROGRAMVBO_H
#define SPECTROGRAMVBO_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <boost/shared_ptr.hpp>
#include <GpuCpuData.h>

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

private:
    Vbo(const Vbo &b);
    Vbo& operator=(const Vbo &b);

    GLuint _vbo;
};


template<typename T>
class MappedVbo
{
public:
    MappedVbo( pVbo vbo, cudaExtent numberOfElements )
    :   data(0, make_cudaExtent(0,0,0)),
        _vbo(vbo)
    {
        void* g_data;
        cudaGLMapBufferObject((void**)&g_data, *_vbo);
        data = GpuCpuData<T>( g_data, numberOfElements, GpuCpuVoidData::CudaGlobal, true );
    }

    ~MappedVbo() {
        cudaGLUnmapBufferObject(*_vbo);
    }

    GpuCpuData<T> data;

private:
    pVbo _vbo;
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

    void draw(SpectrogramRenderer* renderer);
    void draw_directMode( );
private:
    Spectrogram* _spectrogram;

    pVbo _height;
    pVbo _slope;
};


#endif // SPECTROGRAMVBO_H
