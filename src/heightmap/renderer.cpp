#include <cstdio>

// gpusmisc/OpenGL
#include "gl.h"

// sonicawe
#include "heightmap/renderer.h"
#include "heightmap/collection.h"
#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "sawe/configuration.h"
#include "sawe/nonblockingmessagebox.h"
#include "signal/operation.h"
#include "render/renderaxes.h"
#include "render/renderfrustum.h"

// gpumisc
#include <float.h>
#include "GlException.h"
#include "computationkernel.h"
#include "glPushContext.h"
#include "TaskTimer.h"
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

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

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
    _invalid_frustum(true),
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


void Renderer::
        setSize( unsigned w, unsigned h)
{
    _render_block.setSize (w, h);
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

    setSize(2,2);
    beginVboRendering();
    endVboRendering();

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
    _invalid_frustum = true;
    _render_block.clearCaches();
}


Reference Renderer::
        findRefAtCurrentZoomLevel( Heightmap::Position p )
{
    //Position max_ss = collection->max_sample_size();
    Reference ref = read1(collection)->entireHeightmap();
    BlockLayout bc = read1(collection)->block_layout ();

    // The first 'ref' will be a super-ref containing all other refs, thus
    // containing 'p' too. This while-loop zooms in on a ref containing
    // 'p' with enough details.

    // 'p' is assumed to be valid to start with. Ff they're not valid
    // this algorithm will choose some ref along the border closest to the
    // point 'p'.

    while (true)
    {
        LevelOfDetal lod = testLod(ref);

        Region r = RegionFactory(bc)(ref);

        switch(lod)
        {
        case Lod_NeedBetterF:
            if ((r.a.scale+r.b.scale)/2 > p.scale)
                ref = ref.bottom();
            else
                ref = ref.top();
            break;

        case Lod_NeedBetterT:
            if ((r.a.time+r.b.time)/2 > p.time)
                ref = ref.left();
            else
                ref = ref.right();
            break;

        default:
            return ref;
        }
    }
}

/**
  Note: the parameter scaley is used by RenderView to go seamlessly from 3D to 2D.
  This is different from the 'attribute' Renderer::y_scale which is used to change the
  height of the mountains.
  */
void Renderer::draw( float scaley )
{
    if (!collection)
        return;

    if (!read1(collection)->visualization_params ()->transform_desc())
        return;

    GlException_CHECK_ERROR();

    TIME_RENDERER TaskTimer tt("Rendering scaletime plot");
    if (NotInitialized == _initialized) init();
    if (Initialized != _initialized) return;

    _invalid_frustum = true;

    if ((_draw_flat = .001 > scaley))
        setSize(2,2),
        scaley = 0.001;
    else
    {
        BlockLayout block_size = read1(collection)->block_layout ();
        setSize( block_size.texels_per_row ()/_mesh_fraction_width,
                 block_size.texels_per_column ()/_mesh_fraction_height );
    }

    render_settings.last_ysize = scaley;
    render_settings.drawn_blocks = 0;

    glPushMatrixContext mc(GL_MODELVIEW);

    //Position mss = collection->max_sample_size();
    //Reference ref = collection->findReference(Position(0,0), mss);
    Reference ref = read1(collection)->entireHeightmap();

    gl_projection.update();
    _frustum_clip.update (0, 0);

    glScalef(1, _draw_flat ? 0 : scaley, 1);

    beginVboRendering();

    if (!renderChildrenSpectrogramRef(ref))
        renderSpectrogramRef( ref );

    endVboRendering();

    GlException_CHECK_ERROR();
}


void Renderer::beginVboRendering()
{
    BlockLayout block_size = read1(collection)->block_layout ();
    _render_block.beginVboRendering (block_size);
}


void Renderer::endVboRendering() {
    _render_block.endVboRendering ();
}


void Renderer::renderSpectrogramRef( Reference ref )
{
    pBlock block = write1 (collection)->getBlock( ref );

    if (!_render_block.renderBlock(block, _draw_flat)) {
        float y = _frustum_clip.projectionPlane[1]*.05;
        _render_block.renderBlockError(block->block_layout (), block->getRegion (), y);
    }

    render_settings.drawn_blocks++;
}


Renderer::LevelOfDetal Renderer::testLod( Reference ref )
{
    BlockLayout bl = read1(collection)->block_layout ();
    VisualizationParams::ConstPtr vp = read1(collection)->visualization_params ();

    float timePixels, scalePixels;
    if (!computePixelsPerUnit( ref, timePixels, scalePixels ))
        return Lod_Invalid;

    if(0) if (-10==ref.log2_samples_size[0] && -8==ref.log2_samples_size[1]) {
        fprintf(stdout, "Ref (%d,%d)\t%g\t%g\n", ref.block_index[0], ref.block_index[1], timePixels,scalePixels);
        fflush(stdout);
    }

    GLdouble needBetterF, needBetterT;

    if (0==scalePixels)
        needBetterF = 1.01;
    else
        needBetterF = scalePixels / (_redundancy*bl.texels_per_column ());
    if (0==timePixels)
        needBetterT = 1.01;
    else
        needBetterT = timePixels / (_redundancy*bl.texels_per_row ());

    if (!ReferenceInfo(ref.top(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS) &&
        !ReferenceInfo(ref.bottom(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS))
        needBetterF = 0;

    if (!ReferenceInfo(ref.left(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighT))
        needBetterT = 0;

    if ( needBetterF > needBetterT && needBetterF > 1 )
        return Lod_NeedBetterF;

    else if ( needBetterT > 1 )
        return Lod_NeedBetterT;

    else
        return Lod_Ok;
}

bool Renderer::renderChildrenSpectrogramRef( Reference ref )
{
    BlockLayout bl = read1(collection)->block_layout ();
    VisualizationParams::ConstPtr vp = read1(collection)->visualization_params ();

    TIME_RENDERER_BLOCKS TaskTimer tt(boost::format("%s")
          % ReferenceInfo(ref, bl, vp));

    LevelOfDetal lod = testLod( ref );
    switch(lod) {
    case Lod_NeedBetterF:
        renderChildrenSpectrogramRef( ref.bottom() );
        renderChildrenSpectrogramRef( ref.top() );
        break;
    case Lod_NeedBetterT:
        renderChildrenSpectrogramRef( ref.left() );
        if (ReferenceInfo(ref.right (), bl, vp)
                .boundsCheck(ReferenceInfo::BoundsCheck_OutT))
            renderChildrenSpectrogramRef( ref.right() );
        break;
    case Lod_Ok:
        renderSpectrogramRef( ref );
        break;
    case Lod_Invalid: // ref is not within the current view frustum
        return false;
    }

    return true;
}


static void printl(const char* str, const std::vector<GLvector>& l) {
    fprintf(stdout,"%s (%lu)\n",str,(unsigned long)l.size());
    for (unsigned i=0; i<l.size(); i++) {
        fprintf(stdout,"  %g\t%g\t%g\n",l[i][0],l[i][1],l[i][2]);
    }
    fflush(stdout);
}


/**
  @arg ref See timePixels and scalePixels
  @arg timePixels Estimated longest line of pixels along time axis within ref measured in pixels
  @arg scalePixels Estimated longest line of pixels along scale axis within ref measured in pixels
  */
bool Renderer::
        computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels )
{
    Region r = RegionFactory ( read1(collection)->block_layout () )(ref);
    const Position p[2] = { r.a, r.b };

    float y[]={0, float(_frustum_clip.projectionPlane[1]*.5)};
    for (unsigned i=0; i<sizeof(y)/sizeof(y[0]); ++i)
    {
        GLvector corner[]=
        {
            GLvector( p[0].time, y[i], p[0].scale),
            GLvector( p[0].time, y[i], p[1].scale),
            GLvector( p[1].time, y[i], p[1].scale),
            GLvector( p[1].time, y[i], p[0].scale)
        };

        GLvector closest_i;
        std::vector<GLvector> clippedCorners = _frustum_clip.clipFrustum(corner, closest_i); // about 10 us
        if (clippedCorners.empty ())
            continue;

        GLvector::T
                timePerPixel = 0,
                freqPerPixel = 0;

        gl_projection.computeUnitsPerPixel( closest_i, timePerPixel, freqPerPixel );

        // time/scalepixels is approximately the number of pixels in ref along the time/scale axis
        timePixels = (p[1].time - p[0].time)/timePerPixel;
        scalePixels = (p[1].scale - p[0].scale)/freqPerPixel;

        return true;
    }

    return false;
}


void Renderer::
        computeUnitsPerPixel( GLvector p, GLvector::T& timePerPixel, GLvector::T& scalePerPixel )
{
    gl_projection.computeUnitsPerPixel (p, timePerPixel, scalePerPixel);
}


template<typename T>
void swap( T& x, T& y) {
    x = x + y;
    y = x - y;
    x = x - y;
}


void Renderer::
        drawAxes( float T )
{
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


template<typename T> void glVertex3v( const T* );

template<> void glVertex3v( const GLdouble* t ) {    glVertex3dv(t); }
template<>  void glVertex3v( const GLfloat* t )  {    glVertex3fv(t); }

void Renderer::
        drawFrustum()
{
    Render::RenderFrustum(render_settings, clippedFrustum).drawFrustum();
}

} // namespace Heightmap
