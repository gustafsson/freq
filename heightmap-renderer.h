#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

#include <sstream>
#include "heightmap-collection.h"
#ifndef __APPLE__
  #include <GL/gl.h>
#else
  #include <OpenGL/gl.h>
#endif
#include "heightmap-glblock.h"
#include <tmatrix.h>

class DisplayWidget;

typedef tvector<3,GLdouble> GLvector;

namespace Heightmap {

class Renderer
{
public:
    enum ColorMode {
        ColorMode_Rainbow = 0,
        ColorMode_Grayscale = 1
    };

    Renderer( Collection* collection, DisplayWidget* _tempToRemove );

    Collection* collection() { return _collection; }

    void draw( float scaley );
    void drawAxes( float T );
    void drawFrustum( float alpha=0.25f );

    bool draw_piano;
    bool draw_hz;
    GLvector camera;

    bool draw_height_lines;
    ColorMode color_mode;
    float y_scale;
private:
    std::vector<GLvector> clippedFrustum;
    Collection* _collection;
    DisplayWidget* _tempToRemove;
    GLuint _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    bool _draw_flat;
    float _redundancy;

    unsigned _drawn_blocks;

    friend class Heightmap::GlBlock;

    void init();
    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);

    void beginVboRendering();
    void endVboRendering();

    bool renderSpectrogramRef( Reference ref );
    bool renderChildrenSpectrogramRef( Reference ref );
    void renderParentSpectrogramRef( Reference ref );
    bool computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels );
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
