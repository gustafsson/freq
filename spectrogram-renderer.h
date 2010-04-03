#ifndef SPECTROGRAMRENDERER_H
#define SPECTROGRAMRENDERER_H
#include <sstream>
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

static std::string _shaderBaseDir;
class SpectrogramRenderer
{
public:
    SpectrogramRenderer( pSpectrogram spectrogram );
    static void setShaderBaseDir(std::string shaderBaseDir){ _shaderBaseDir = shaderBaseDir; printf("Shaderbasedir: %s\n", _shaderBaseDir.c_str());}

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


#endif // SPECTROGRAMRENDERER_H
