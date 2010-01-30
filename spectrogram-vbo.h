#ifndef SPECTROGRAMVBO_H
#define SPECTROGRAMVBO_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <boost/shared_ptr.hpp>
#include <GpuCpuData.h>
#include <stdio.h>

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
    :   _vbo(vbo)
    {
        void* g_data;
        GLuint n = *_vbo;
        cudaGLMapBufferObject((void**)&g_data, *_vbo);

        cudaPitchedPtr cpp = {g_data, sizeof(T)*numberOfElements.width, sizeof(T)*numberOfElements.width, numberOfElements.height };
        data.reset( new GpuCpuData<T>( &cpp, numberOfElements, GpuCpuVoidData::CudaGlobal, true ));
    }

    ~MappedVbo() {
        cudaGLUnmapBufferObject(*_vbo);
    }

    boost::shared_ptr<GpuCpuData<T> > data;

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
