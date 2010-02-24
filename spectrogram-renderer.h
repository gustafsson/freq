#ifndef SPECTROGRAMRENDERER_H
#define SPECTROGRAMRENDERER_H

#include "spectrogram.h"
#ifdef _MSC_VER
#include <windows.h>
#endif
#ifndef __APPLE__
  #include <GL/gl.h>
#else
  #include <OpenGL/gl.h>
#endif
#include "spectrogram-vbo.h"

class SpectrogramRenderer
{
public:
    SpectrogramRenderer( pSpectrogram spectrogram );

    pSpectrogram spectrogram() { return _spectrogram; }

    void draw();
private:
    pSpectrogram _spectrogram;
    GLuint _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    float _redundancy;

    unsigned _drawn_blocks;

    friend class SpectrogramVbo;

    void init();
    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);

    void beginVboRendering();
    void endVboRendering();

    bool renderSpectrogramRef( Spectrogram::Reference ref );
    bool renderChildrenSpectrogramRef( Spectrogram::Reference ref );
    void renderParentSpectrogramRef( Spectrogram::Reference ref );
    bool computePixelsPerUnit( Spectrogram::Reference ref, float& timePixels, float& scalePixels );
};


#endif // SPECTROGRAMRENDERER_H
