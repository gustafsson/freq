#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

#include <sstream>
#include "heightmap/collection.h"
#ifndef __APPLE__
  #include <GL/gl.h>
#else
  #include <OpenGL/gl.h>
#endif
#include "heightmap/glblock.h"
#include <tmatrix.h>
#include <GlTexture.h>

typedef tvector<3,GLdouble> GLvector;

namespace Heightmap {

    template<typename f>
    GLvector gluProject(tvector<3,f> obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);

    template<typename f>
    GLvector gluUnProject(tvector<3,f> win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);

    template<typename f>
    GLvector gluProject(tvector<3,f> obj, bool *r=0);

    GLvector gluUnProject(GLvector win, bool *r=0);

class Renderer
{
public:
    enum ColorMode {
        ColorMode_Rainbow = 0,
        ColorMode_Grayscale = 1,
        ColorMode_FixedColor = 2
    };

    Renderer( Collection* collection );

    Reference findRefAtCurrentZoomLevel( float t, float s );
    Collection* collection;

    void draw( float scaley );
    void drawAxes( float T );
    void drawFrustum( float alpha=0.25f );

    bool draw_piano;
    bool draw_hz;
    GLvector camera;

    bool draw_height_lines;
    ColorMode color_mode;
    float4 fixed_color;
    float y_scale;
    float last_ysize;

    void init();

private:
    enum LevelOfDetal {
        Lod_NeedBetterF,
        Lod_NeedBetterT,
        Lod_Ok,
        Lod_Invalid
    };

    std::vector<GLvector> clippedFrustum;
    GLuint _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    bool _draw_flat;
    float _redundancy;

    boost::scoped_ptr<GlTexture> colorTexture;

    unsigned _drawn_blocks;

    friend class Heightmap::GlBlock;

    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);
    void createColorTexture(unsigned N);

    void beginVboRendering();
    void endVboRendering();

    bool renderSpectrogramRef( Reference ref );
    LevelOfDetal testLod( Reference ref );
    bool renderChildrenSpectrogramRef( Reference ref );
    void renderParentSpectrogramRef( Reference ref );
    bool computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels );
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
