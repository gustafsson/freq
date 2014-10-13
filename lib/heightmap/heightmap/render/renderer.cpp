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

// Qt
#include <QSettings>

#ifdef _MSC_VER
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
        glProjection         gl_projection,
        Render::RenderBlock* render_block)
    :
      collection(collection),
      render_settings(render_settings),
      gl_projection(gl_projection),
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
        createMissingBlocks(R);

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
        mesh_fraction_width*=4;
        mesh_fraction_height*=4;
#endif

        render_block->setSize (
                 block_size.texels_per_row ()/mesh_fraction_width,
                 block_size.texels_per_column ()/mesh_fraction_height );
    }

    render_settings.last_ysize = scaley;
    render_settings.drawn_blocks = 0;

    const auto& v = gl_projection.viewport;
    glViewport (v[0], v[1], v[2], v[3]);

    const auto& c = render_settings.clear_color;
    glClearColor (c[0], c[1], c[2], c[3]);
}


Render::RenderSet::references_t Renderer::
        getRenderSet(float L)
{
    BlockLayout bl                   = collection.read ()->block_layout ();
    Reference ref                    = collection.read ()->entireHeightmap();
    VisualizationParams::const_ptr vp = collection.read ()->visualization_params ();
    Render::RenderInfo render_info(&gl_projection, bl, vp, render_settings.redundancy);
    Render::RenderSet::references_t R = Render::RenderSet(&render_info, L).computeRenderSet( ref );

    LOG_REFERENCES_TO_RENDER {
        TaskInfo("Rendering %d blocks", R.size());
        for ( auto const& r : R)
            TaskInfo(boost::format("%s") % ReferenceInfo(r, bl, vp));
    }

    return R;
}


void Renderer::
        createMissingBlocks(const Render::RenderSet::references_t& R)
{
    collection.raw ()->createMissingBlocks (R);
}


void Renderer::
        drawBlocks(const Render::RenderSet::references_t& R)
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawBlocks");

    Render::RenderSet::references_t failed;

    {
        auto collection = this->collection.read ();
        int frame_number = collection->frame_number ();
        BlockLayout bl = collection->block_layout ();
        collection.unlock ();

        // Copy the block list
        auto cache = this->collection.raw ()->cache ()->clone ();

        Render::RenderBlock::Renderer block_renderer(render_block, bl, gl_projection);

        for (const Reference& r : R)
        {
            auto i = cache.find(r);
            if (i != cache.end())
            {
                pBlock block = i->second;
                block_renderer.renderBlock(block);
                block->frame_number_last_used = frame_number;
                render_settings.drawn_blocks++;
            }
            else
            {
                // Indicate unavailable blocks by not drawing the surface but only a wireframe.
                failed.insert(r);
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

    BlockLayout bl = collection.read ()->block_layout ();
    RegionFactory region(bl);

    Render::RenderRegion rr(gl_projection);

    for (const Reference& r : R)
        rr.render(region(r), drawcross);
}


} // namespace Render
} // namespace Heightmap
