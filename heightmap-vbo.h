#ifndef SPECTROGRAMVBO_H
#define SPECTROGRAMVBO_H

#include <cuda_runtime.h>
#ifdef _MSC_VER
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include <GpuCpuData.h>
#include <stdio.h>
#include <TaskTimer.h>
#include <mappedvbo.h>

typedef boost::shared_ptr<class SpectrogramVbo> pSpectrogramVbo;
typedef boost::shared_ptr<class SpectrogramRenderer> pSpectrogramRenderer;
typedef boost::shared_ptr<class Spectrogram> pSpectrogram;


GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

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
