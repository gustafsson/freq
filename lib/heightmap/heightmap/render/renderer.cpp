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
#include "renderaxes.h"
#include "renderfrustum.h"
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

#define DISPLAY_REFERENCES
//#define DISPLAY_REFERENCES if(0)

using namespace std;

namespace Heightmap {
namespace Render {

Renderer::Renderer()
    :
    _render_block( &render_settings )
{
}


unsigned Renderer::
        trianglesPerBlock()
{
    return _render_block.trianglesPerBlock ();
}


bool Renderer::
        isInitialized()
{
    return _render_block.isInitialized ();
}


void Renderer::init()
{
    _render_block.init();
}


void Renderer::
        clearCaches()
{
    _render_block.clearCaches();
}


Reference Renderer::
        findRefAtCurrentZoomLevel( Heightmap::Position p )
{
    Reference entireHeightmap = collection.read ()->entireHeightmap();
    BlockLayout bl = collection.read ()->block_layout();
    VisualizationParams::const_ptr vp = collection.read ()->visualization_params();
    Render::RenderInfo ri(&gl_projection, bl, vp, render_settings.redundancy);
    Reference r = Render::RenderSet(&ri, 0).computeRefAt (p, entireHeightmap);
    return r;
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
    if (!_render_block.isInitialized ())
        return;

    glPushMatrixContext mc(GL_MODELVIEW);
    setupGlStates(scaley);

    float yscalelimit = render_settings.drawcrosseswhen0 ? 0.0004f : 0.f;
    bool draw = render_settings.y_scale > yscalelimit;

    if (draw)
    {
        Render::RenderSet::references_t R = getRenderSet(L);
        createMissingBlocks(R);
        drawBlocks(R);

        DISPLAY_REFERENCES drawReferences(R, false);
    }
    else
    {
        Render::RenderSet::references_t R = getRenderSet(L);
        drawReferences(R);
    }

    GlException_CHECK_ERROR();
}


void Renderer::
        setupGlStates(float scaley)
{
    if ((render_settings.draw_flat = .001 > scaley))
    {
        _render_block.setSize (2, 2);
        scaley = 0.001;
    }
    else
    {
        BlockLayout block_size = collection.read ()->block_layout ();

        unsigned mesh_fraction_width, mesh_fraction_height;
        mesh_fraction_width = mesh_fraction_height = 1 << (int)(render_settings.redundancy*.5f);
        // mesh resolution equal to texel resolution if mesh_fraction_width == 1 and mesh_fraction_height == 1

        _render_block.setSize (
                 block_size.texels_per_row ()/mesh_fraction_width,
                 block_size.texels_per_column ()/mesh_fraction_height );
    }

    render_settings.last_ysize = scaley;
    render_settings.drawn_blocks = 0;

    glScalef(1, render_settings.draw_flat ? 0 : scaley, 1);
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

        Render::RenderBlock::Renderer block_renderer(&_render_block, bl);

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

    for (const Reference& r : R)
        Render::RenderRegion(region(r)).render(drawcross);
}


void Renderer::
        drawAxes( float T )
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawAxes");

    FreqAxis display_scale = collection.read ()->visualization_params ()->display_scale();

    Render::RenderAxes(
                render_settings,
                &gl_projection,
                display_scale
                ).drawAxes(T);
}


} // namespace Render
} // namespace Heightmap
