#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

// Heightmap namespace
#include "collection.h"
#include "glblock.h"

// gpumisc
#include <tmatrix.h>
#include <GlTexture.h>

// std
#include <sstream>

//typedef tvector<3,GLdouble> GLvector;
typedef tvector<3,GLfloat> GLvector;
typedef tvector<3,GLfloat> GLvectorF;

namespace Heightmap {

    GLvector gluProject(GLvectorF obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);
    GLvector gluUnProject(GLvectorF win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);

class Renderer
{
public:
    enum ColorMode {
        ColorMode_Rainbow = 0,
        ColorMode_Grayscale = 1,
        ColorMode_FixedColor = 2
    };

    Renderer( Collection* collection );

    Reference findRefAtCurrentZoomLevel( Heightmap::Position p );
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
    unsigned drawn_blocks;

    void init();

    GLdouble modelview_matrix[16], projection_matrix[16];
    GLint viewport_matrix[4];

    GLvector gluProject(GLvectorF obj, bool *r=0);
    GLvector gluUnProject(GLvectorF win, bool *r=0);

    void frustumMinMaxT( float& min_t, float& max_t);
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
    unsigned _vbo_size;
    pVbo _mesh_position;
    GLuint _shader_prog;
    bool _initialized;
    bool _draw_flat;
    float _redundancy;
    bool _invalid_frustum;
    GLvector projectionPlane, projectionNormal, // for clipFrustum
        rightPlane, rightNormal,
        leftPlane, leftNormal,
        topPlane, topNormal,
        bottomPlane, bottomNormal;

    boost::scoped_ptr<GlTexture> colorTexture;

    friend class Heightmap::GlBlock;

    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(unsigned w, unsigned h);
    void createMeshPositionVBO(unsigned w, unsigned h);
    void createColorTexture(unsigned N);

    void beginVboRendering();
    void endVboRendering();

    void renderSpectrogramRef( Reference ref );
    LevelOfDetal testLod( Reference ref );
    bool renderChildrenSpectrogramRef( Reference ref );
    void renderParentSpectrogramRef( Reference ref );
    bool computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels );

    std::vector<GLvector> clipFrustum( GLvector corner[4], GLvector &closest_i, float w=0, float h=0 );
    std::vector<GLvector> clipFrustum( std::vector<GLvector> l, GLvector &closest_i, float w=0, float h=0 );
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
