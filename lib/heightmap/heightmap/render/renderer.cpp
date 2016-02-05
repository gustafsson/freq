#include <cstdio>

// gpusmisc/OpenGL
#include "gl.h"

// sonicawe
#include "heightmap/render/renderer.h"
#include "heightmap/collection.h"
#include "heightmap/block.h"
#include "heightmap/reference_hash.h"
#include "heightmap/render/renderregion.h"
#include "signal/operation.h"
#include "renderinfo.h"

// gpumisc
#include <float.h>
#include "GlException.h"
#include "computationkernel.h"
#include "glPushContext.h"
#include "tasktimer.h"
#include "GlTexture.h"
#include "glgroupmarker.h"

// Qt
#include <QSettings>

#ifdef _WIN32
#include "msc_stdc.h"
#endif

//#define TIME_RENDERER
#define TIME_RENDERER if(0)

//#define TIME_RENDERER_DETAILS
#define TIME_RENDERER_DETAILS if(0)

//#define LOG_REFERENCES_TO_RENDER
#define LOG_REFERENCES_TO_RENDER if(0)

//#define DISPLAY_REFERENCES
#define DISPLAY_REFERENCES if(0)

using namespace std;

namespace Heightmap {
namespace Render {

Renderer::Renderer(
        Collection::ptr      collection,
        RenderSettings&      render_settings,
        glProjecter          gl_projecter,
        Render::RenderBlock* render_block)
    :
      collection(collection),
      render_settings(render_settings),
      gl_projecter(gl_projecter),
      render_block(render_block)
{

}


void Renderer::
        draw( float scaley, float L )
{
    if (!collection)
        return;

    if (!collection.read ()->visualization_params ()->detail_info())
        return;

    GlException_CHECK_ERROR();

    TIME_RENDERER TaskTimer tt("Rendering scaletime plot");
    if (!render_block->isInitialized ())
        return;

    Render::RenderSet::references_t R = getRenderSet(L);
    float yscalelimit = render_settings.drawcrosseswhen0 ? 0.0004f : 0.f;
    bool draw = render_settings.y_scale > yscalelimit;
    if (draw)
        prepareBlocks(R);

    setupGlStates(scaley);

    if (draw)
        drawBlocks(R);
    else
        drawReferences(R);

    if (draw) DISPLAY_REFERENCES drawReferences(R, false);

    GlException_CHECK_ERROR();
}


void Renderer::
        setupGlStates(float scaley)
{
    if ((render_settings.draw_flat = .001 > scaley))
    {
        render_block->setSize (2, 2);
        scaley = 0.001;
    }
    else
    {
        BlockLayout block_size = collection.read ()->block_layout ();

        unsigned mesh_fraction_width, mesh_fraction_height;
        mesh_fraction_width = mesh_fraction_height = 1 << (int)(render_settings.redundancy*.5f);
        // mesh resolution equal to texel resolution if mesh_fraction_width == 1 and mesh_fraction_height == 1

#ifdef GL_ES_VERSION_2_0
        // too many vertices
//        mesh_fraction_width*=4;
//        mesh_fraction_height*=4;
#endif

        // Match each vertex to each texel. The width measured in texels is 'visible_texels_per_row'
        // but the border texels are half texels so the total number of vertices is 'visible_texels_per_row+1'
        render_block->setSize (
                 (block_size.visible_texels_per_row ()+1)/mesh_fraction_width,
                 (block_size.visible_texels_per_column ()+1)/mesh_fraction_height );
    }

    render_settings.last_ysize = scaley;
    render_settings.drawn_blocks = 0;

    const auto& v = gl_projecter.viewport;
    glViewport (v[0], v[1], v[2], v[3]);
}


Render::RenderSet::references_t Renderer::
        getRenderSet(float L)
{
    auto cr = collection.read ();
    BlockLayout bl                   = cr->block_layout ();
    Reference ref                    = cr->entireHeightmap();
    VisualizationParams::const_ptr vp = cr->visualization_params ();
    cr.unlock ();
    Render::RenderInfo render_info(&gl_projecter, bl, vp, render_settings.redundancy);
    Render::RenderSet::references_t R = Render::RenderSet(&render_info, L).computeRenderSet( ref );

    LOG_REFERENCES_TO_RENDER {
        TaskInfo("Rendering %d blocks", R.size());
        for ( auto const& r : R)
            TaskInfo(boost::format("%s") % ReferenceInfo(r.first, bl, vp));
    }

    return R;
}


void Renderer::
        prepareBlocks (const Render::RenderSet::references_t& R)
{
    bool use_ota = render_settings.y_normalize > 0;
    bool use_mipmap = true;

    // check all blocks if mipmap settings have changed, or if mipmap is needed for a new texture
    int wanted_mipmap = use_mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR;
    {
        BlockCache::cache_t cache = Collection::cache (collection)->clone ();
        for (const auto& v : cache)
        {
            bool domipmap = false;
            if (use_ota != (bool)v.second->texture_ota())
            {
                v.second->enableOta (use_ota);
                domipmap |= use_ota;
            }

            int f = v.second->texture ()->getMinFilter ();
            if (wanted_mipmap != f) // if changed
            {
                v.second->texture ()->bindTexture ();
                v.second->texture ()->setMinFilter (wanted_mipmap);
                domipmap |= use_mipmap;
            }

            if (domipmap)
                v.second->generateMipmap (); // will check min filter and only regenerate if used
        }
    }

    collection->prepareBlocks (R);
}


void Renderer::
        drawBlocks(const Render::RenderSet::references_t& R)
{
    GlGroupMarker gpm("DrawBlocks");
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawBlocks");

    Render::RenderSet::references_t failed;

    {
        BlockLayout bl = collection.read ()->block_layout ();

        // Copy the block list
        auto cache = Collection::cache (this->collection)->clone ();

        Render::RenderBlock::Renderer block_renderer(render_block, bl, gl_projecter);

        for (const auto& v : R)
        {
            const Reference& r = v.first;
            auto i = cache.find(r);
            if (i != cache.end())
            {
                pBlock block = i->second;
                block_renderer.renderBlock(block, v.second);
                render_settings.drawn_blocks++;
            }
            else
            {
                // Indicate unavailable blocks by not drawing the surface but only a wireframe.
                failed.insert(v);
            }
        }

    } // finish Render::RenderBlock::Renderer

    drawReferences(failed);
}


void Renderer::
        drawReferences(const Render::RenderSet::references_t& R, bool drawcross)
{
    if (R.empty ())
        return;

    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawReferences");
    GlGroupMarker gpm("DrawReferences");

    BlockLayout bl = collection.read ()->block_layout ();
    RegionFactory region(bl);

    Render::RenderRegion* rr = render_block->renderRegion ();

    for (const auto& r : R)
        rr->render(gl_projecter, region.getVisible (r.first), drawcross);
}


} // namespace Render
} // namespace Heightmap
