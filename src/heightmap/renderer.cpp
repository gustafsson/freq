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
    _mesh_index_buffer(0),
    _mesh_width(0),
    _mesh_height(0),
    _mesh_fraction_width(1),
    _mesh_fraction_height(1),
    _shader_prog(0),
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
    _drawcrosseswhen0( Sawe::Configuration::version().empty() ),
    _frustum_clip( &gl_projection, &render_settings.left_handed_axes ),
    _color_texture_colors( (RenderSettings::ColorMode)-1 )
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
    return (_mesh_width-1) * (_mesh_height-1) * 2;
}


void Renderer::setSize( unsigned w, unsigned h)
{
    if (w == _mesh_width && h ==_mesh_height)
        return;

    createMeshIndexBuffer(w, h);
    createMeshPositionVBO(w, h);
}

// create index buffer for rendering quad mesh
void Renderer::createMeshIndexBuffer(int w, int h)
{
    GlException_CHECK_ERROR();

    // create index buffer
    if (_mesh_index_buffer)
        glDeleteBuffersARB(1, &_mesh_index_buffer);

    // edge dropout to eliminate visible glitches
    if (w>2) w+=2;
    if (h>2) h+=2;

    _mesh_width = w;
    _mesh_height = h;

    _vbo_size = ((w*2)+4)*(h-1);

    std::vector<BLOCKindexType> indicesdata(_vbo_size);
    BLOCKindexType *indices = &indicesdata[0];
    if (indices) for(int y=0; y<h-1; y++) {
        *indices++ = y*w;
        for(int x=0; x<w; x++) {
            *indices++ = y*w+x;
            *indices++ = (y+1)*w+x;
        }
        // start new strip with degenerate triangle
        *indices++ = (y+1)*w+(w-1);
        *indices++ = (y+1)*w;
        *indices++ = (y+1)*w;
    }

    glGenBuffers(1, &_mesh_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);

    // fill with indices for rendering mesh as triangle strips
    GlException_SAFE_CALL( glBufferData(GL_ELEMENT_ARRAY_BUFFER, _vbo_size*sizeof(BLOCKindexType), &indicesdata[0], GL_STATIC_DRAW) );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}

// create fixed vertex buffer to store mesh vertices
void Renderer::createMeshPositionVBO(int w, int h)
{
    int y1 = 0, x1 = 0, y2 = h, x2 = w;

    // edge dropout to eliminate visible glitches
    if (w>2) x1--, x2++;
    if (h>2) y1--, y2++;

    std::vector<float> posdata( (x2-x1)*(y2-y1)*4 );
    float *pos = &posdata[0];

    for(int y=y1; y<y2; y++) {
        for(int x=x1; x<x2; x++) {
            float u = x / (float) (w-1);
            float v = y / (float) (h-1);
            *pos++ = u;
            *pos++ = 0.0f;
            *pos++ = v;
            *pos++ = 1.0f;
        }
    }

    _mesh_position.reset( new Vbo( (w+2)*(h+2)*4*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW, &posdata[0] ));

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

    // load shader
    if (render_settings.vertex_texture)
        _shader_prog = loadGLSLProgram(":/shaders/heightmap.vert", ":/shaders/heightmap.frag");
    else
        _shader_prog = loadGLSLProgram(":/shaders/heightmap_noshadow.vert", ":/shaders/heightmap.frag");

    if (0 == _shader_prog)
        return;

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
    _mesh_width = 0;
    _mesh_height = 0;
    _initialized = NotInitialized;
    _mesh_position.reset();
    glDeleteProgram(_shader_prog);
    _shader_prog = 0;
    _invalid_frustum = true;
    _colorTexture.reset();
    _color_texture_colors = (RenderSettings::ColorMode)-1;
}


tvector<4,float> mix(tvector<4,float> a, tvector<4,float> b, float f)
{
    return a*(1-f) + b*f;
}

tvector<4,float> getWavelengthColorCompute( float wavelengthScalar, RenderSettings::ColorMode scheme ) {
    tvector<4,float> spectrum[12];
    int count = 0;
    spectrum[0] = tvector<4,float>( 0, 0, 0, 0 );

    switch (scheme)
    {
    case RenderSettings::ColorMode_GreenRed:
        spectrum[0] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[1] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[4] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[5] = tvector<4,float>( 1, 1, 0, 0 ),
        spectrum[6] = tvector<4,float>( 1, 1, 0, 0 ),
        spectrum[7] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[8] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[9] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[10] = tvector<4,float>( -0.5, 0, 0, 0 ); // dark line, almost black
        spectrum[11] = tvector<4,float>( 0.75, 0, 0, 0 ); // dark red when over the top
        count = 11;
        break;
    case RenderSettings::ColorMode_GreenWhite:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        spectrum[0] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[1] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[3] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[4] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[5] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[6] = tvector<4,float>( 1, 1, 1, 0 );
        //spectrum[7] = tvector<4,float>( -0.5, -0.5, -0.5, 0 ); // dark line, almost black
        spectrum[7] = tvector<4,float>( 1, 1, 1, 0 ); // darker when over the top
        count = 7;
        break;
    case RenderSettings::ColorMode_Green:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        spectrum[0] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[1] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[4] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[5] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[6] = tvector<4,float>( 0, 1, 0, 0 );
        count = 6;
        break;
    case RenderSettings::ColorMode_Grayscale:
        break;
    case RenderSettings::ColorMode_BlackGrayscale:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        break;
    default:
        /* for white background */
        float a = 1/255.f;
        // rainbow http://en.wikipedia.org/wiki/Rainbow#Spectrum
        spectrum[0] = tvector<4,float>( 1, 0, 0, 0 ), // red
        spectrum[1] = tvector<4,float>( 148*a, 0, 211*a, 0 ), // violet
        spectrum[2] = tvector<4,float>( 148*a, 0, 211*a, 0 ), // violet
        spectrum[3] = tvector<4,float>( 75*a, 0, 130*a, 0 ), // indigo
        spectrum[4] = tvector<4,float>( 0, 0, 1, 0 ), // blue
        spectrum[5] = tvector<4,float>( 0, 0.5, 0, 0 ), // green
        spectrum[6] = tvector<4,float>( 1, 1, 0, 0 ), // yellow
        spectrum[7] = tvector<4,float>( 1, 0.5, 0, 0 ), // orange
        spectrum[8] = tvector<4,float>( 1, 0, 0, 0 ), // red
        spectrum[9] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[10] = tvector<4,float>( -0.5, 0, 0, 0 ); // dark line, almost black
        spectrum[11] = tvector<4,float>( 0.75, 0, 0, 0 ); // dark red when over the top
        count = 11;
//        spectrum[0] = tvector<4,float>( 1, 0, 1, 0 ),
//        spectrum[1] = tvector<4,float>( 0, 0, 1, 0 ),
//        spectrum[2] = tvector<4,float>( 0, 1, 1, 0 ),
//        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 ),
//        spectrum[4] = tvector<4,float>( 1, 1, 0, 0 ),
//        spectrum[5] = tvector<4,float>( 1, 0, 1, 0 ),
//        spectrum[6] = tvector<4,float>( 1, 0, 0, 0 );
//        spectrum[7] = tvector<4,float>( 1, 0, 0, 0 );
//        spectrum[8] = tvector<4,float>( 0, 0, 0, 0 );
//        spectrum[9] = tvector<4,float>( 0.5, 0, 0, 0 );
//        count = 9;//sizeof(spectrum)/sizeof(spectrum[0])-1;

        /* for black background
            { 0, 0, 0 },
            { 1, 0, 1 },
            { 0, 0, 1 },
            { 0, 1, 1 },
            { 0, 1, 0 },
            { 1, 1, 0 },
            { 1, 0, 0 }}; */
        break;
    }

    if (wavelengthScalar<0)
        return tvector<4,float>( 1, 1, 1, 1 );

    float f = float(count)*wavelengthScalar;
    int i1 = int(floor(max(0.f, min(f-1.f, float(count)))));
    int i2 = int(floor(max(0.f, min(f, float(count)))));
    int i3 = int(floor(max(0.f, min(f+1.f, float(count)))));
    int i4 = int(floor(max(0.f, min(f+2.f, float(count)))));
    float t = (f-float(i2))*0.5;
    float s = 0.5 + t;

    tvector<4,float> rgb = mix(spectrum[i1], spectrum[i3], s) + mix(spectrum[i2], spectrum[i4], t);
    rgb = rgb * 0.5;
    //TaskInfo("%g %g %g: %g %g %g %g", f, t, s, rgb[0], rgb[1], rgb[2], rgb[3]);
    return rgb;
}

void Renderer::createColorTexture(unsigned N) {
    if (_color_texture_colors == render_settings.color_mode && _colorTexture && _colorTexture->getWidth()==N)
        return;

    _color_texture_colors = render_settings.color_mode;

    std::vector<tvector<4,float> > texture(N);
    for (unsigned i=0; i<N; ++i) {
        texture[i] = getWavelengthColorCompute( i/(float)(N-1), _color_texture_colors );
    }
    _colorTexture.reset( new GlTexture(N,1, GL_RGBA, GL_RGBA, GL_FLOAT, &texture[0]));

    render_settings.clear_color = getWavelengthColorCompute( -1.f, _color_texture_colors );
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
    GlException_CHECK_ERROR();
    //unsigned meshW = collection->samples_per_block();
    //unsigned meshH = collection->scales_per_block();

    createColorTexture(24); // These will be linearly interpolated when rendering, so a high resolution texture is not needed
    glActiveTexture(GL_TEXTURE2);
    _colorTexture->bindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    glUseProgram(_shader_prog);

    // TODO check if this takes any time
    {   // Set default uniform variables parameters for the vertex and pixel shader
        TIME_RENDERER_BLOCKS TaskTimer tt("Setting shader parameters");
        GLuint uniVertText0, uniVertText1, uniVertText2, uniColorTextureFactor, uniFixedColor, uniClearColor, uniContourPlot, uniYScale, uniScaleTex, uniOffsTex;

        uniVertText0 = glGetUniformLocation(_shader_prog, "tex");
        glUniform1i(uniVertText0, 0); // GL_TEXTURE0

        uniVertText1 = glGetUniformLocation(_shader_prog, "tex_nearest");
        glUniform1i(uniVertText1, 1); // GL_TEXTURE1

        uniVertText2 = glGetUniformLocation(_shader_prog, "tex_color");
        glUniform1i(uniVertText2, 2); // GL_TEXTURE2

        uniFixedColor = glGetUniformLocation(_shader_prog, "fixedColor");
        switch (render_settings.color_mode)
        {
        case RenderSettings::ColorMode_Grayscale:
            glUniform4f(uniFixedColor, 0.f, 0.f, 0.f, 0.f);
            break;
        case RenderSettings::ColorMode_BlackGrayscale:
            glUniform4f(uniFixedColor, 1.f, 1.f, 1.f, 0.f);
            break;
        default:
        {
            tvector<4, float> fixed_color = render_settings.fixed_color;
            glUniform4f(uniFixedColor, fixed_color[0], fixed_color[1], fixed_color[2], fixed_color[3]);
            break;
        }
        }

        uniClearColor = glGetUniformLocation(_shader_prog, "clearColor");
        tvector<4, float> clear_color = render_settings.clear_color;
        glUniform4f(uniClearColor, clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

        uniColorTextureFactor = glGetUniformLocation(_shader_prog, "colorTextureFactor");
        switch(render_settings.color_mode)
        {
        case RenderSettings::ColorMode_Rainbow:
        case RenderSettings::ColorMode_GreenRed:
        case RenderSettings::ColorMode_GreenWhite:
        case RenderSettings::ColorMode_Green:
            glUniform1f(uniColorTextureFactor, 1.f);
            break;
        default:
            glUniform1f(uniColorTextureFactor, 0.f);
            break;
        }

        uniContourPlot = glGetUniformLocation(_shader_prog, "contourPlot");
        glUniform1f(uniContourPlot, render_settings.draw_contour_plot ? 1.f : 0.f );

        uniYScale = glGetUniformLocation(_shader_prog, "yScale");
        glUniform1f(uniYScale, render_settings.y_scale);

        BlockLayout block_size = read1(collection)->block_layout ();
        float
                w = block_size.texels_per_row (),
                h = block_size.texels_per_column ();

        uniScaleTex = glGetUniformLocation(_shader_prog, "scale_tex");
        glUniform2f(uniScaleTex, (w-1.f)/w, (h-1.f)/h);

        uniOffsTex = glGetUniformLocation(_shader_prog, "offset_tex");
        glUniform2f(uniOffsTex, .5f/w, .5f/h);
    }

    glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);

    GlException_CHECK_ERROR();
}

void Renderer::endVboRendering() {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glActiveTexture(GL_TEXTURE2);
    _colorTexture->unbindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}


void Renderer::renderSpectrogramRef( Reference ref )
{
    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();

    Region r = RegionFactory (read1 (collection)->block_layout ()) ( ref );
    glPushMatrixContext mc( GL_MODELVIEW );

    glTranslatef(r.a.time, 0, r.a.scale);
    glScalef(r.time(), 1, r.scale());

    pBlock block = write1 (collection)->getBlock( ref );

    float yscalelimit = _drawcrosseswhen0 ? 0.0004f : 0.f;
    if (0!=block.get() && render_settings.y_scale > yscalelimit) {
        if (0 /* direct rendering */ )
            ;//block->glblock->draw_directMode();
        else if (1 /* vbo */ )
            block->glblock->draw( _vbo_size, _draw_flat ? GlBlock::HeightMode_Flat : render_settings.vertex_texture ? GlBlock::HeightMode_VertexTexture : GlBlock::HeightMode_VertexBuffer);

    } else if ( 0 == "render red warning cross" || render_settings.y_scale < yscalelimit) {
        endVboRendering();
        // getBlock would try to find something else if the requested block
        // wasn't readily available.

        // If getBlock fails, we're most likely out of memory. Indicate this
        // silently by not drawing the surface but only a wireframe.

        glPushAttribContext attribs;

        glDisable(GL_TEXTURE_2D);
        glDisable(GL_BLEND);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHTING);
        glBindTexture(GL_TEXTURE_2D, 0);
        glColor4f( 0.8f, 0.2f, 0.2f, 0.5f );
        glLineWidth(2);

        glBegin(GL_LINE_STRIP);
            glVertex3f( 0, 0, 0 );
            glVertex3f( 1, 0, 1 );
            glVertex3f( 1, 0, 0 );
            glVertex3f( 0, 0, 1 );
            glVertex3f( 0, 0, 0 );
            glVertex3f( 1, 0, 0 );
            glVertex3f( 1, 0, 1 );
            glVertex3f( 0, 0, 1 );
        glEnd();
        float y = _frustum_clip.projectionPlane[1]*.05;
        glColor4f( 0.2f, 0.8f, 0.8f, 0.5f );
        glBegin(GL_LINE_STRIP);
            glVertex3f( 0, y, 0 );
            glVertex3f( 1, y, 1 );
            glVertex3f( 1, y, 0 );
            glVertex3f( 0, y, 1 );
            glVertex3f( 0, y, 0 );
            glVertex3f( 1, y, 0 );
            glVertex3f( 1, y, 1 );
            glVertex3f( 0, y, 1 );
        glEnd();
        beginVboRendering();
    }

    render_settings.drawn_blocks++;

    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();
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
        std::vector<GLvector> clippedCorners = _frustum_clip.clipFrustum(corner, closest_i);
        if (0) if (-10==ref.log2_samples_size[0] && -8==ref.log2_samples_size[1])
        {
            printl("Clipped corners",clippedCorners);
            printf("closest_i %g\t%g\t%g\n", closest_i[0], closest_i[1], closest_i[2]);
        }
        if (0==clippedCorners.size())
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
