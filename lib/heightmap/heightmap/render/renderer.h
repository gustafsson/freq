#ifndef HEIGHTMAP_RENDER_RENDERER_H
#define HEIGHTMAP_RENDER_RENDERER_H

// Heightmap namespace
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/collection.h"
#include "rendersettings.h"
#include "renderblock.h"
#include "renderset.h"

// gpumisc
#include "shared_state.h"

class GlTexture;

namespace Heightmap {
namespace Render {

/**
 * @brief The Renderer class is a shallow class.
 *
 * It doesn't produce/own/maintain/create/release any state.
 */
class Renderer
{
public:
    Renderer(shared_state<Collection>        collection,
             RenderSettings&                 render_settings,
             glProjecter                     gl_projecter,
             Render::RenderBlock*            render_block);

    /**
      Note: the parameter scaley is used by RenderView to go seamlessly from 3D to 2D.
      This is different from the 'attribute' Renderer::y_scale which is used to change the
      height of the mountains.
      */
    void draw( float scaley, float T );

private:
    shared_state<Collection>        collection;
    RenderSettings&                 render_settings;
    const glProjecter               gl_projecter;
    Render::RenderBlock*            render_block;

    void setupGlStates(float scaley);
    Render::RenderSet::references_t getRenderSet(float L);
    void prepareBlocks(const Render::RenderSet::references_t& R);
    void drawBlocks(const Render::RenderSet::references_t& R);
    void drawReferences(const Render::RenderSet::references_t& R, bool drawcross=true);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERER_H
