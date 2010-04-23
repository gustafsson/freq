#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

#include <sstream>
#include "heightmap-collection.h"
//#ifdef _MSC_VER
//#include <windows.h>
//#endif
#ifndef __APPLE__
  #include <GL/gl.h>
#else
  #include <OpenGL/gl.h>
#endif
#include "heightmap-glblock.h"

class DisplayWidget;

namespace Heightmap {

class Renderer
{
public:
    Renderer( pCollection collection, DisplayWidget* _tempToRemove );

    pCollection collection() { return _collection; }

    void draw();
    void drawAxes();

    bool draw_piano;
    bool draw_hz;
private:
    pCollection _collection;
    DisplayWidget* _tempToRemove;
    GLuint _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    float _redundancy;

    float _fewest_pixles_per_unit;
    Reference _fewest_pixles_per_unit_ref;

    unsigned _drawn_blocks;

    friend class Heightmap::GlBlock;

    void init();
    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);

    void beginVboRendering();
    void endVboRendering();

    bool renderSpectrogramRef( Reference ref, bool* finished_ref );
    bool renderChildrenSpectrogramRef( Reference ref );
    void renderParentSpectrogramRef( Reference ref );
    bool computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels );
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
