#ifndef HEIGHTMAP_RENDER_RENDERER_H
#define HEIGHTMAP_RENDER_RENDERER_H

// Heightmap namespace
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/collection.h"
#include "rendersettings.h"
#include "frustumclip.h"
#include "renderblock.h"
#include "renderset.h"

// gpumisc
#include "shared_state.h"

// std
#include <vector>

// boost
#include <boost/shared_ptr.hpp>

class GlTexture;

namespace Heightmap {
namespace Render {

class Renderer
{
public:
    Renderer();

    shared_state<Collection>        collection;
    RenderSettings                  render_settings;
    glProjection                    gl_projection;

    Reference findRefAtCurrentZoomLevel( Heightmap::Position p );

    /**
      Note: the parameter scaley is used by RenderView to go seamlessly from 3D to 2D.
      This is different from the 'attribute' Renderer::y_scale which is used to change the
      height of the mountains.
      */
    void draw( float scaley, float T );
    void drawAxes( float T );

    unsigned trianglesPerBlock();
    void setSize( unsigned w, unsigned h );
    bool isInitialized();
    void init();

    void clearCaches();

private:

    enum InitializedLevel {
        NotInitialized,
        Initialized,
        InitializationFailed
    };

    InitializedLevel _initialized;
    Render::RenderBlock _render_block;

    void setupGlStates(float scaley);
    Render::RenderSet::references_t getRenderSet(float L);
    void createMissingBlocks(const Render::RenderSet::references_t& R);
    void drawBlocks(const Render::RenderSet::references_t& R);
    void drawReferences(const Render::RenderSet::references_t& R, bool drawcross=true);
};
typedef boost::shared_ptr<Renderer> pRenderer;

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERER_H
