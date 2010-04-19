#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

#include <sstream>
#include "spectrogram.h"
//#ifdef _MSC_VER
//#include <windows.h>
//#endif
#ifndef __APPLE__
  #include <GL/gl.h>
#else
  #include <OpenGL/gl.h>
#endif
#include "heightmap-vbo.h"

namespace Heightmap {

class Renderer
{
public:
    Renderer( pSpectrogram spectrogram );

    pSpectrogram spectrogram() { return _spectrogram; }

    void draw();
    void drawAxes();

    bool draw_piano;
    bool draw_hz;
private:
    pSpectrogram _spectrogram;
    GLuint _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    float _redundancy;

    float _fewest_pixles_per_unit;
    Spectrogram::Reference _fewest_pixles_per_unit_ref;

    unsigned _drawn_blocks;

    friend class SpectrogramVbo;

    void init();
    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);

    void beginVboRendering();
    void endVboRendering();

    bool renderSpectrogramRef( Spectrogram::Reference ref, bool* finished_ref );
    bool renderChildrenSpectrogramRef( Spectrogram::Reference ref );
    void renderParentSpectrogramRef( Spectrogram::Reference ref );
    bool computePixelsPerUnit( Spectrogram::Reference ref, float& timePixels, float& scalePixels );
};

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
