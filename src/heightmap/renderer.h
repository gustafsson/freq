#ifndef HEIGHTMAPRENDERER_H
#define HEIGHTMAPRENDERER_H

// Heightmap namespace
#include "reference.h"
#include "position.h"
#include "rendersettings.h"
#include "render/frustumclip.h"

// gpumisc
#include "volatileptr.h"

// std
#include <vector>

// boost
#include <boost/shared_ptr.hpp>

class GlTexture;

class Vbo;
typedef boost::shared_ptr<Vbo> pVbo;

namespace Heightmap {

    class Collection;

class Renderer
{
public:
    Renderer();

    VolatilePtr<Collection>::Ptr    collection;
    RenderSettings                  render_settings;
    glProjection                    gl_projection;

    Reference findRefAtCurrentZoomLevel( Heightmap::Position p );

    void draw( float scaley );
    void drawAxes( float T );
    void drawFrustum();

    void setFractionSize( unsigned divW=1, unsigned divH=1);
    bool fullMeshResolution();
    unsigned trianglesPerBlock();
    bool isInitialized();
    void init();

    float redundancy();
    void redundancy(float value);

    void clearCaches();

private:
    enum LevelOfDetal {
        Lod_NeedBetterF,
        Lod_NeedBetterT,
        Lod_Ok,
        Lod_Invalid
    };

    enum InitializedLevel {
        NotInitialized,
        Initialized,
        InitializationFailed
    };

    unsigned _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    unsigned _mesh_fraction_width;
    unsigned _mesh_fraction_height;
    unsigned _vbo_size;
    pVbo _mesh_position;
    unsigned _shader_prog;
    InitializedLevel _initialized;
    bool _draw_flat;
    float _redundancy;
    bool _invalid_frustum;
    bool _drawcrosseswhen0;
    Render::FrustumClip _frustum_clip;
    std::vector<GLvector> clippedFrustum;

    RenderSettings::ColorMode _color_texture_colors;
    boost::shared_ptr<GlTexture> _colorTexture;

    void setSize( unsigned w, unsigned h);
    void createMeshIndexBuffer(int w, int h);
    void createMeshPositionVBO(int w, int h);
    void createColorTexture(unsigned N);

    void beginVboRendering();
    void endVboRendering();

    void renderSpectrogramRef( Reference ref );
    LevelOfDetal testLod( Reference ref );
    bool renderChildrenSpectrogramRef( Reference ref );
    bool computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels );
    void computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel );
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Heightmap

#endif // HEIGHTMAPRENDERER_H
