#include <cstdio>

// gpusmisc/OpenGL
#include "gl.h"

// sonicawe
#include "heightmap/renderer.h"
#include "heightmap/collection.h"
#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "heightmap/reference_hash.h"
#include "heightmap/blocks/chunkmerger.h"
#include "heightmap/render/renderregion.h"
#include "sawe/configuration.h"
#include "sawe/nonblockingmessagebox.h"
#include "signal/operation.h"
#include "render/renderaxes.h"
#include "render/renderfrustum.h"
#include "render/renderinfo.h"

// gpumisc
#include <float.h>
#include "GlException.h"
#include "computationkernel.h"
#include "glPushContext.h"
#include "tasktimer.h"
#include "GlTexture.h"

// boost
#include <boost/foreach.hpp>

// Qt
#include <QSettings>

#ifdef _MSC_VER
#include "msc_stdc.h"
#endif

//#define TIME_RENDERER
#define TIME_RENDERER if(0)

//#define TIME_RENDERER_DETAILS
#define TIME_RENDERER_DETAILS if(0)

using namespace std;

namespace Heightmap {


Renderer::Renderer()
    :
    _initialized(NotInitialized),
    _draw_flat(false),
    /*
     reasoning about the default _redundancy value.
     The thing about Sonic AWE is a good visualization. In this there is value
     booth in smooth navigation and high resolution. As the navigation is fast
     on new computers even with high resolution we set this value to give most
     people a good first impression. For people with older computers it's
     possible to suggest that they lower the resolution for faster navigation.

     This could be done through a dropdownnotification if plain rendering
     takes too long.
     */
    _redundancy(1.0f), // 1 means every pixel gets at least one texel (and vertex), 10 means every 10th pixel gets its own vertex, default=2
    _frustum_clip( &gl_projection, &render_settings.left_handed_axes ),
    _render_block( &render_settings ),
    _mesh_fraction_width(1),
    _mesh_fraction_height(1)
{
    _mesh_fraction_width = _mesh_fraction_height = 1 << (int)(_redundancy*.5f);
}


void Renderer::
        setFractionSize( unsigned divW, unsigned divH)
{
    _mesh_fraction_width = divW;
    _mesh_fraction_height = divH;
}


bool Renderer::
        fullMeshResolution()
{
    return _mesh_fraction_height == 1 && _mesh_fraction_width == 1;
}


unsigned Renderer::
        trianglesPerBlock()
{
    return _render_block.trianglesPerBlock ();
}


bool Renderer::
        isInitialized()
{
    return Initialized == _initialized;
}


void Renderer::init()
{
    if (NotInitialized != _initialized)
        return;

    // assume failure unless we reach the end of this method
    _initialized = InitializationFailed;

    TaskTimer tt("Initializing OpenGL");

    GlException_CHECK_ERROR();

#ifdef USE_CUDA
    int cudaDevices=0;
    CudaException_SAFE_CALL( cudaGetDeviceCount( &cudaDevices) );
    if (0 == cudaDevices ) {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't find any \"cuda capable\" device",
                "This version of Sonic AWE requires a graphics card that supports CUDA and no such graphics card was found.\n\n"
                "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this." );

        // fail
        return;
    }
#endif

#ifndef __APPLE__ // glewInit is not needed on Mac
    if (0 != glewInit() ) {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't properly setup graphics",
                "Sonic AWE failed to setup required graphics hardware.\n\n"
                "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this.",

                "Couldn't initialize \"glew\"");

        // fail
        return;
    }
#endif

    // verify necessary OpenGL extensions
    const char* glversion = (const char*)glGetString(GL_VERSION);
    string glversionstring(glversion);
    stringstream versionreader(glversionstring);
    int gl_major=0, gl_minor=0;
    char dummy;
    versionreader >> gl_major >> dummy >> gl_minor;

    //TaskInfo("OpenGL version %d.%d (%s)", gl_major, gl_minor, glversion);

    if ((1 > gl_major )
        || ( 1 == gl_major && 4 > gl_minor ))
    {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't properly setup graphics",
                "Sonic AWE requires a graphics driver that supports OpenGL 2.0 and no such graphics driver was found.\n\n"
                "If you think this messge is an error, please file this as a bug report at muchdifferent.com to help us fix this.");

        // fail
        return;
    }

    const char* exstensions[] = {
        "GL_ARB_vertex_buffer_object",
        "GL_ARB_pixel_buffer_object",
        "",
        "GL_ARB_texture_float"
    };

    bool required_extension = true;
    const char* all_extensions = (const char*)glGetString(GL_EXTENSIONS);
    //TaskInfo("Checking extensions %s", all_extensions);
    for (unsigned i=0; i < sizeof(exstensions)/sizeof(exstensions[0]); ++i)
    {
        if (0 == strlen(exstensions[i]))
        {
            required_extension = false;
            continue;
        }


        bool hasExtension = 0 != strstr(all_extensions, exstensions[i]);
        if (!hasExtension)
            TaskInfo("%s %s extension %s",
                     hasExtension?"Found":"Couldn't find",
                     required_extension?"required":"preferred",
                     exstensions[i]);

        if (hasExtension)
            continue;

        std::stringstream err;
        std::stringstream details;

        err << "Sonic AWE can't properly setup graphics. ";
        if (required_extension)
        {
            err << "Sonic AWE requires features that couldn't be found on your graphics card.";
            details << "Sonic AWE requires a graphics card that supports '" << exstensions[i] << "'";
        }
        else
        {
            bool warn_expected_opengl = QSettings().value("warn_expected_opengl", true).toBool();
            if (!warn_expected_opengl)
                continue;
             QSettings().setValue("warn_expected_opengl", false);

            err << "Sonic AWE works better with features that couldn't be found on your graphics card. "
                << "However, Sonic AWE might still run. Click OK to try.";
            details << "Sonic AWE works better with a graphics card that supports '" << exstensions[i] << "'";
        }

        err << endl << endl << "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this.";

        Sawe::NonblockingMessageBox::show(
                required_extension?QMessageBox::Critical : QMessageBox::Warning,
                "Couldn't properly setup graphics",
                err.str().c_str(),
                details.str().c_str());

        if (required_extension)
            return;
    }

    _render_block.init();

    _render_block.setSize (2, 2);
    drawBlocks(Render::RenderSet::references_t());

    _initialized=Initialized;

    GlException_CHECK_ERROR();
}


float Renderer::
        redundancy()
{
    return _redundancy;
}


void Renderer::
        redundancy(float value)
{
    _redundancy = value;
}


void Renderer::
        clearCaches()
{
    _initialized = NotInitialized;
    _render_block.clearCaches();
}


Reference Renderer::
        findRefAtCurrentZoomLevel( Heightmap::Position p )
{
    Reference entireHeightmap = read1(collection)->entireHeightmap();
    BlockLayout bl = read1(collection)->block_layout();
    VisualizationParams::ConstPtr vp = read1(collection)->visualization_params();
    Render::RenderInfo ri(&gl_projection, bl, vp, &_frustum_clip, _redundancy);
    Reference r = Render::RenderSet(&ri, 0).computeRefAt (p, entireHeightmap);
    return r;
}


void Renderer::
        draw( float scaley, float L )
{
    if (!collection)
        return;

    if (!read1(collection)->visualization_params ()->transform_desc())
        return;

    GlException_CHECK_ERROR();

    TIME_RENDERER TaskTimer tt("Rendering scaletime plot");
    if (NotInitialized == _initialized) init();
    if (Initialized != _initialized) return;

    glPushMatrixContext mc(GL_MODELVIEW);
    setupGlStates(scaley);

    float yscalelimit = render_settings.drawcrosseswhen0 ? 0.0004f : 0.f;
    bool draw = render_settings.y_scale > yscalelimit;

    if (draw)
    {
        Render::RenderSet::references_t R = getRenderSet(L);
        createMissingBlocks(R);
        updateTextures(R);
        drawBlocks(R);
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
        BlockLayout block_size = read1(collection)->block_layout ();
        _render_block.setSize (
                 block_size.texels_per_row ()/_mesh_fraction_width,
                 block_size.texels_per_column ()/_mesh_fraction_height );
    }

    render_settings.last_ysize = scaley;
    render_settings.drawn_blocks = 0;


    gl_projection.update();
    _frustum_clip.update (0, 0);

    glScalef(1, render_settings.draw_flat ? 0 : scaley, 1);
}


Render::RenderSet::references_t Renderer::
        getRenderSet(float L)
{
    BlockLayout bl                   = read1(collection)->block_layout ();
    Reference ref                    = read1(collection)->entireHeightmap();
    VisualizationParams::ConstPtr vp = read1(collection)->visualization_params ();
    Render::RenderInfo render_info(&gl_projection, bl, vp, &_frustum_clip, _redundancy);
    Render::RenderSet::references_t R = Render::RenderSet(&render_info, L).computeRenderSet( ref );

    return R;
}


void Renderer::
        createMissingBlocks(const Render::RenderSet::references_t& R)
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::createMissingBlocks");


    Render::RenderSet::references_t missing;

    {
        BlockCache::ReadPtr cache( read1(collection)->cache () );
        BOOST_FOREACH(const Reference& r, R) {
            if (!cache->probe(r))
                missing.insert (r);
        }
    }

    if (!missing.empty ())
    {
        Collection::ReadPtr collectionp(collection);

        BOOST_FOREACH(const Reference& r, missing) {
            // Create blocks
            collectionp->getBlock (r);
        }
    }
}


void Renderer::
        updateTextures(const Render::RenderSet::references_t& R)
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::updateTextures");

    BlockCache::ReadPtr cache( read1(collection)->cache () );

    for (const Reference& r : R)
    {
        if (pBlock block = cache->probe( r )) {
            block->update_glblock_data ();
            block->glblock->update_texture (
                        render_settings.draw_flat
                        ?
                            GlBlock::HeightMode_Flat
                          : render_settings.vertex_texture
                            ?
                                GlBlock::HeightMode_VertexTexture
                              : GlBlock::HeightMode_VertexBuffer );
        }
    }
}


void Renderer::
        drawBlocks(const Render::RenderSet::references_t& R)
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawBlocks");

    Render::RenderSet::references_t failed;

    {
        int frame_number = read1(collection)->frame_number ();
        BlockLayout bl = read1(collection)->block_layout ();
        BlockCache::Ptr cache = read1(collection)->cache ();

        BlockCache::WritePtr cachep( cache );
        Render::RenderBlock::Renderer block_renderer(&_render_block, bl);

        BOOST_FOREACH(const Reference& r, R)
        {
            if (pBlock block = cachep->find( r ))
            {
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
        drawReferences(const Render::RenderSet::references_t& R)
{
    if (R.empty ())
        return;

    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawReferences");

    BlockLayout bl = read1(collection)->block_layout ();
    RegionFactory region(bl);

    BOOST_FOREACH(const Reference& r, R) {
        Render::RenderRegion(region(r)).render();
    }
}


void Renderer::
        drawAxes( float T )
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawAxes");

    Tfr::FreqAxis display_scale = read1(collection)->visualization_params ()->display_scale();

    Render::RenderAxes ra(
                render_settings,
                &gl_projection,
                &_frustum_clip,
                display_scale
                );
    ra.drawAxes(T);
    clippedFrustum = ra.getClippedFrustum ();
}


void Renderer::
        drawFrustum()
{
    TIME_RENDERER_DETAILS TaskTimer tt("Renderer::drawFrustum");

    Render::RenderFrustum(render_settings, clippedFrustum).drawFrustum();
}

} // namespace Heightmap
