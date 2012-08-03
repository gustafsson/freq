#include <cstdio>

// gpusmisc/OpenGL
#include <gl.h>

// glut
#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif

// sonicawe
#include "heightmap/renderer.h"
#include "heightmap/collection.h"
#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "sawe/configuration.h"
#include "sawe/nonblockingmessagebox.h"

// gpumisc
#include <float.h>
#include <GlException.h>
#include <computationkernel.h>
#include <glPushContext.h>
#include <TaskTimer.h>
#include <GlTexture.h>

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


Renderer::Renderer( Collection* collection )
:   collection(collection),
    draw_piano(false),
    draw_hz(true),
    draw_t(true),
    draw_cursor_marker(false),
    draw_axis_at0(0),
    camera(0,0,0),
    draw_contour_plot(false),
    color_mode( ColorMode_Rainbow ),
    fixed_color( 1,0,0,1 ),
    clear_color( 1,1,1,1 ),
    y_scale( 1 ),
    last_ysize( 1 ),
    last_axes_length( 0 ),
    drawn_blocks(0),
    left_handed_axes(true),
    vertex_texture(false),
    _mesh_index_buffer(0),
    _mesh_width(0),
    _mesh_height(0),
    _mesh_fraction_width(1),
    _mesh_fraction_height(1),
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
    _color_texture_colors( (ColorMode)-1 )
{
    memset(modelview_matrix, 0, sizeof(modelview_matrix));
    memset(projection_matrix, 0, sizeof(projection_matrix));
    memset(viewport_matrix, 0, sizeof(viewport_matrix));

    _mesh_fraction_width = _mesh_fraction_height = 1 << (int)(_redundancy*.5f);

    // Using glut for drawing fonts, so glutInit must be called.
    static int c=0;
    if (0==c)
    {
		// run glutinit once per process
#ifdef _WIN32
		c = 1;
		char* dummy="dummy\0";
		glutInit(&c,&dummy);
#elif !defined(__APPLE__)
        glutInit(&c,0);
        c = 1;
#endif
    }
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
void Renderer::createMeshIndexBuffer(unsigned w, unsigned h)
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
    if (indices) for(unsigned y=0; y<h-1; y++) {
        *indices++ = y*w;
        for(unsigned x=0; x<w; x++) {
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
void Renderer::createMeshPositionVBO(unsigned w, unsigned h)
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

typedef tvector<4,GLdouble> GLvector4;
typedef tmatrix<4,GLdouble> GLmatrix;

GLvector gluProject(GLvector obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r) {
//    //gluProject does this, (win - screenspace).dot() < 1e18
//    tmatrix<4, double> modelmatrix(model);
//    tmatrix<4, double> projmatrix(proj);
//    tvector<4, double> obj4(obj[0], obj[1], obj[2], 1);
//    tvector<4, double> proj4 = projmatrix*modelmatrix*obj4;
//    tvector<3, double> screennorm(proj4[0]/proj4[3], proj4[1]/proj4[3], proj4[2]/proj4[3]);
//    GLvector screenspace(view[0] + (screennorm[0] + 1.0)*0.5*view[2],
//                                   view[1] + (screennorm[1] + 1.0)*0.5*view[3],
//                                   0.5 + 0.5*screennorm[2]);
//    if (r)
//        *r = 0 != proj4[3];
//    return screenspace;

    GLvector win;
    bool s = (GLU_TRUE == ::gluProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]));
    if (r)
        *r = s;

    return win;
}

GLvector gluUnProject(GLvector win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r) {
    GLdouble obj0=0, obj1=0, obj2=0;
    bool s = (GLU_TRUE == ::gluUnProject(win[0], win[1], win[2], model, proj, view, &obj0, &obj1, &obj2));
    if(r) *r=s;
    return GLvector(obj0, obj1, obj2);
}


/* distance along normal, a negative distance means obj is in front of plane */
static float distanceToPlane( GLvector obj, const GLvector& plane, const GLvector& normal ) {
    return (plane-obj)%normal;
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

    TaskInfo("OpenGL version %d.%d (%s)", gl_major, gl_minor, glversion);

    if ((1 > gl_major )
        || ( 1 == gl_major && 4 > gl_minor ))
    {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't properly setup graphics",
                "Sonic AWE requires a graphics card that supports OpenGL 2.0 and no such graphics card was found.\n\n"
                "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this.");

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
    TaskInfo("Checking extensions %s", all_extensions);
    for (unsigned i=0; i < sizeof(exstensions)/sizeof(exstensions[0]); ++i)
    {
        if (0 == strlen(exstensions[i]))
        {
            required_extension = false;
            continue;
        }


        bool hasExtension = 0 != strstr(all_extensions, exstensions[i]);
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
    if (vertex_texture)
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


GLvector Renderer::
        gluProject(GLvector obj, bool *r)
{
    return Heightmap::gluProject(obj, modelview_matrix, projection_matrix, viewport_matrix, r);
}


GLvector Renderer::
        gluUnProject(GLvector win, bool *r)
{
    return Heightmap::gluUnProject(win, modelview_matrix, projection_matrix, viewport_matrix, r);
}


void Renderer::
        frustumMinMaxT( float& min_t, float& max_t )
{
    max_t = 0;
    min_t = FLT_MAX;

    BOOST_FOREACH( GLvector v, clippedFrustum)
    {
        if (max_t < v[0])
            max_t = v[0];
        if (min_t > v[0])
            min_t = v[0];
    }
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


tvector<4,float> mix(tvector<4,float> a, tvector<4,float> b, float f)
{
    return a*(1-f) + b*f;
}

tvector<4,float> getWavelengthColorCompute( float wavelengthScalar, Renderer::ColorMode scheme ) {
    tvector<4,float> spectrum[12];
    int count = 0;
    switch (scheme)
    {
    case Renderer::ColorMode_GreenRed:
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
    case Renderer::ColorMode_GreenWhite:
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
    if (_color_texture_colors == color_mode && colorTexture->getWidth()==N)
        return;

    _color_texture_colors = color_mode;

    std::vector<tvector<4,float> > texture(N);
    for (unsigned i=0; i<N; ++i) {
        texture[i] = getWavelengthColorCompute( i/(float)(N-1), _color_texture_colors );
    }
    colorTexture.reset( new GlTexture(N,1, GL_RGBA, GL_RGBA, GL_FLOAT, &texture[0]));

    clear_color = getWavelengthColorCompute( -1.f, _color_texture_colors );
}

Reference Renderer::
        findRefAtCurrentZoomLevel( Heightmap::Position p )
{
    //Position max_ss = collection->max_sample_size();
    Reference ref = collection->entireHeightmap();

    // The first 'ref' will be a super-ref containing all other refs, thus
    // containing 'p' too. This while-loop zooms in on a ref containing
    // 'p' with enough details.

    // 'p' is assumed to be valid to start with. Ff they're not valid
    // this algorithm will choose some ref along the border closest to the
    // point 'p'.

    while (true)
    {
        LevelOfDetal lod = testLod(ref);

        Region r = ref.getRegion();

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
    GlException_CHECK_ERROR();

    TIME_RENDERER TaskTimer tt("Rendering scaletime plot");
    if (NotInitialized == _initialized) init();
    if (Initialized != _initialized) return;

    _invalid_frustum = true;

    if ((_draw_flat = .001 > scaley))
        setSize(2,2),
        scaley = 0.001;
    else
        setSize( collection->samples_per_block()/_mesh_fraction_width, collection->scales_per_block()/_mesh_fraction_height );

    last_ysize = scaley;
    drawn_blocks = 0;

    glPushMatrixContext mc(GL_MODELVIEW);

    //Position mss = collection->max_sample_size();
    //Reference ref = collection->findReference(Position(0,0), mss);
    Reference ref = collection->entireHeightmap();

    glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
    glGetIntegerv(GL_VIEWPORT, viewport_matrix);

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
    colorTexture->bindTexture2D();
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
        if (color_mode == ColorMode_Grayscale)
            glUniform4f(uniFixedColor, 1.f, 1.f, 1.f, 1.f);
        else
            glUniform4f(uniFixedColor, fixed_color[0], fixed_color[1], fixed_color[2], fixed_color[3]);

        uniClearColor = glGetUniformLocation(_shader_prog, "clearColor");
        glUniform4f(uniClearColor, clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

        uniColorTextureFactor = glGetUniformLocation(_shader_prog, "colorTextureFactor");
        switch(color_mode)
        {
        case ColorMode_Rainbow:
        case ColorMode_GreenRed:
        case ColorMode_GreenWhite:
            glUniform1f(uniColorTextureFactor, 1.f);
            break;
        default:
            glUniform1f(uniColorTextureFactor, 0.f);
            break;
        }

        uniContourPlot = glGetUniformLocation(_shader_prog, "contourPlot");
        glUniform1f(uniContourPlot, draw_contour_plot ? 1.f : 0.f );

        uniYScale = glGetUniformLocation(_shader_prog, "yScale");
        glUniform1f(uniYScale, y_scale);

        float
                w = collection->samples_per_block(),
                h = collection->scales_per_block();

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
    colorTexture->unbindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}

void Renderer::renderSpectrogramRef( Reference ref )
{
    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();

    Region r = ref.getRegion();
    glPushMatrixContext mc( GL_MODELVIEW );

    glTranslatef(r.a.time, 0, r.a.scale);
    glScalef(r.time(), 1, r.scale());

    pBlock block = collection->getBlock( ref );

    float yscalelimit = _drawcrosseswhen0 ? 0.0004f : 0.f;
    if (0!=block.get() && y_scale > yscalelimit) {
        if (0 /* direct rendering */ )
            ;//block->glblock->draw_directMode();
        else if (1 /* vbo */ )
            block->glblock->draw( _vbo_size, _draw_flat ? GlBlock::HeightMode_Flat : vertex_texture ? GlBlock::HeightMode_VertexTexture : GlBlock::HeightMode_VertexBuffer);

    } else if ( 0 == "render red warning cross" || y_scale < yscalelimit) {
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
        float y = projectionPlane[1]*.5;
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

    drawn_blocks++;

    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();
}


Renderer::LevelOfDetal Renderer::testLod( Reference ref )
{
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
        needBetterF = scalePixels / (_redundancy*collection->scales_per_block());
    if (0==timePixels)
        needBetterT = 1.01;
    else
        needBetterT = timePixels / (_redundancy*collection->samples_per_block());

    if (!ref.top().boundsCheck(Reference::BoundsCheck_HighS) && !ref.bottom().boundsCheck(Reference::BoundsCheck_HighS))
        needBetterF = 0;

    if (!ref.left().boundsCheck(Reference::BoundsCheck_HighT))
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
    TIME_RENDERER_BLOCKS TaskTimer tt("%s", ref.toString().c_str());

    LevelOfDetal lod = testLod( ref );
    switch(lod) {
    case Lod_NeedBetterF:
        renderChildrenSpectrogramRef( ref.bottom() );
        renderChildrenSpectrogramRef( ref.top() );
        break;
    case Lod_NeedBetterT:
        renderChildrenSpectrogramRef( ref.left() );
        if (ref.right().boundsCheck(Reference::BoundsCheck_OutT))
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

void Renderer::renderParentSpectrogramRef( Reference ref )
{
    // Assume that ref has already been drawn, draw sibblings, and call renderParent again
    renderChildrenSpectrogramRef( ref.sibbling1() );
    renderChildrenSpectrogramRef( ref.sibbling2() );
    renderChildrenSpectrogramRef( ref.sibbling3() );

    if (!ref.parent().tooLarge() )
        renderParentSpectrogramRef( ref.parent() );
}

// the normal does not need to be normalized
static GLvector planeIntersection( GLvector const& pt1, GLvector const& pt2, float &s, GLvector const& plane, GLvector const& normal ) {
    GLvector dir = pt2-pt1;

    s = ((plane-pt1)%normal)/(dir % normal);
    GLvector p = pt1 + dir * s;

//    float v = (p-plane ) % normal;
//    fprintf(stdout,"p[2] = %g, v = %g\n", p[2], v);
//    fflush(stdout);
    return p;
}


static void clipPlane( std::vector<GLvector>& p, const GLvector& p0, const GLvector& n )
{
    if (p.empty())
        return;

    unsigned i;

    GLvector const* a, * b = &p[p.size()-1];
    bool a_side, b_side = (p0-*b)%n < 0;
    for (i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = (p0-*b)%n < 0;

        if (a_side != b_side )
        {
            GLvector dir = *b-*a;

            // planeIntersection
            float s = ((p0-*a)%n)/(dir % n);

            // TODO why [-.1, 1.1]?
            //if (!isnan(s) && -.1 <= s && s <= 1.1)
            if (!isnan(s) && 0 <= s && s <= 1)
            {
                break;
            }
        }
    }

    if (i==p.size())
    {
        if (!b_side)
            p.clear();

        return;
    }

    std::vector<GLvector> r;
    r.reserve(2*p.size());

    b = &p[p.size()-1];
    b_side = (p0-*b)%n < 0;

    for (unsigned i=0; i<p.size(); i++) {
        a = b;
        b = &p[i];

        a_side = b_side;
        b_side = (p0-*b)%n <0;

        if (a_side)
            r.push_back( *a );

        if (a_side != b_side )
        {
            float s;
            GLvector xy = planeIntersection( *a, *b, s, p0, n );

            //if (!isnan(s) && -.1 <= s && s <= 1.1)
            if (!isnan(s) && 0 <= s && s <= 1)
            {
                r.push_back( xy );
            }
        }
    }

    p = r;
}

static void printl(const char* str, const std::vector<GLvector>& l) {
    fprintf(stdout,"%s (%lu)\n",str,(unsigned long)l.size());
    for (unsigned i=0; i<l.size(); i++) {
        fprintf(stdout,"  %g\t%g\t%g\n",l[i][0],l[i][1],l[i][2]);
    }
    fflush(stdout);
}

/* returns the point on the border of the polygon 'l' that lies closest to 'target' */
static GLvector closestPointOnPoly( const std::vector<GLvector>& l, const GLvector &target)
{
    GLvector r;
    // check if point lies inside
    bool allNeg = true, allPos = true;

    // find point in poly closest to projectionPlane
    float min = FLT_MAX;
    for (unsigned i=0; i<l.size(); i++) {
        float f = (l[i]-target).dot();
        if (f<min) {
            min = f;
            r = l[i];
        }

        GLvector d = (l[(i+1)%l.size()] - l[i]),
                 v = target - l[i];

        if (0==d.dot())
            continue;

        if (d%v < 0) allNeg=false;
        if (d%v > 0) allPos=false;
        float k = d%v / (d.dot());
        if (0<k && k<1) {
            f = (l[i] + d*k-target).dot();
            if (f<min) {
                min = f;
                r = l[i]+d*k;
            }
        }
    }

    if (allNeg || allPos) {
        // point lies within convex polygon, create normal and project to surface
        if (l.size()>2) {
            GLvector n = (l[0]-l[1])^(l[0]-l[2]);
            if (0 != n.dot()) {
                n = n.Normalized();
                r = target + n*distanceToPlane( target, l[0], n );
            }
        }
    }
    return r;
}

std::vector<GLvector> Renderer::
        clipFrustum( std::vector<GLvector> l, GLvector &closest_i, float w, float h )
{
    if (!(0==w && 0==h))
        _invalid_frustum = true;

    if (_invalid_frustum) {
        // this takes about 5 us
        GLint const* const& view = viewport_matrix;

        double z0=.1, z1=.2;
        if (0==w && 0==h)
            _invalid_frustum = false;

        projectionPlane = gluUnProject( GLvector( view[0] + view[2]/2, view[1] + view[3]/2, z0) );
        projectionNormal = (gluUnProject( GLvector( view[0] + view[2]/2, view[1] + view[3]/2, z1) ) - projectionPlane).Normalized();

        rightPlane = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z0) );
        GLvector rightZ = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z1) );
        GLvector rightY = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2+1, z0) );
        rightZ = rightZ - rightPlane;
        rightY = rightY - rightPlane;
        rightNormal = ((rightY)^(rightZ)).Normalized();

        leftPlane = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z0) );
        GLvector leftZ = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z1) );
        GLvector leftY = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2+1, z0) );
        leftNormal = ((leftZ-leftPlane)^(leftY-leftPlane)).Normalized();

        topPlane = gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z0) );
        GLvector topZ = gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z1) );
        GLvector topX = gluUnProject( GLvector( view[0] + view[2]/2+1, view[1] + (1-h)*view[3], z0) );
        topNormal = ((topZ-topPlane)^(topX-topPlane)).Normalized();

        bottomPlane = gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z0) );
        GLvector bottomZ = gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z1) );
        GLvector bottomX = gluUnProject( GLvector( view[0] + view[2]/2+1, view[1]+h*view[3], z0) );
        bottomNormal = ((bottomX-bottomPlane)^(bottomZ-bottomPlane)).Normalized();

        // must make all normals negative because one of the axes is flipped (glScale with a minus sign on the x-axis)
        if (left_handed_axes)
        {
            rightNormal = -rightNormal;
            leftNormal = -leftNormal;
            topNormal = -topNormal;
            bottomNormal = -bottomNormal;
        }

        // Don't bother with projectionNormal?
        projectionNormal = projectionNormal;
    }

    //printl("Start",l);
    // Don't bother with projectionNormal?
    //clipPlane(l, projectionPlane, projectionNormal);
    //printl("Projectionclipped",l);
    clipPlane(l, rightPlane, rightNormal);
    //printl("Right", l);
    clipPlane(l, leftPlane, leftNormal);
    //printl("Left", l);
    clipPlane(l, topPlane, topNormal);
    //printl("Top",l);
    clipPlane(l, bottomPlane, bottomNormal);
    //printl("Bottom",l);
    //printl("Clipped polygon",l);

    closest_i = closestPointOnPoly(l, projectionPlane);
    return l;
}


std::vector<GLvector> Renderer::
        clipFrustum( GLvector corner[4], GLvector &closest_i, float w, float h )
{
    std::vector<GLvector> l;
    l.reserve(4);
    for (unsigned i=0; i<4; i++)
    {
        if (!l.empty() && l.back() == corner[i])
            continue;

        l.push_back( corner[i] );
    }

    return clipFrustum(l, closest_i, w, h);
}

/**
  @arg ref See timePixels and scalePixels
  @arg timePixels Estimated longest line of pixels along time axis within ref measured in pixels
  @arg scalePixels Estimated longest line of pixels along scale axis within ref measured in pixels
  */
bool Renderer::
        computePixelsPerUnit( Reference ref, float& timePixels, float& scalePixels )
{
    Region r = ref.getRegion();
    const Position p[2] = { r.a, r.b };

    float y[]={0, projectionPlane[1]*.5};
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
        std::vector<GLvector> clippedCorners = clipFrustum(corner, closest_i);
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

        computeUnitsPerPixel( closest_i, timePerPixel, freqPerPixel );

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
    // Find units per pixel at point 'p' with glUnProject
    GLvector screen = gluProject( p );
    GLvector screenX=screen, screenY=screen;
    if (screen[0] > viewport_matrix[0] + viewport_matrix[2]/2)
        screenX[0]--;
    else
        screenX[0]++;

    if (screen[1] > viewport_matrix[1] + viewport_matrix[3]/2)
        screenY[1]--;
    else
        screenY[1]++;

    GLvector
            wBase = gluUnProject( screen ),
            w1 = gluUnProject( screenX ),
            w2 = gluUnProject( screenY );

    screen[2]+=1;
    screenX[2]+=1;
    screenY[2]+=1;

    // directions
    GLvector
            dirBase = gluUnProject( screen )-wBase,
            dir1 = gluUnProject( screenX )-w1,
            dir2 = gluUnProject( screenY )-w2;

    // valid projection on xz-plane exists if dir?[1]<0 wBase[1]<0
    GLvector
            xzBase = wBase - dirBase*(wBase[1]/dirBase[1]),
            xz1 = w1 - dir1*(w1[1]/dir1[1]),
            xz2 = w2 - dir2*(w2[1]/dir2[1]);

    // compute {units in xz-plane} per {screen pixel}, that determines the required resolution
    timePerPixel = 0;
    scalePerPixel = 0;

    if (dir1[1] != 0 && dirBase[1] != 0) {
        timePerPixel = max(timePerPixel, fabs(xz1[0]-xzBase[0]));
        scalePerPixel = max(scalePerPixel, fabs(xz1[2]-xzBase[2]));
    }
    if (dir2[1] != 0 && dirBase[1] != 0) {
        timePerPixel = max(timePerPixel, fabs(xz2[0]-xzBase[0]));
        scalePerPixel = max(scalePerPixel, fabs(xz2[2]-xzBase[2]));
    }

    if (0 == timePerPixel)
        timePerPixel = max(fabs(w1[0]-wBase[0]), fabs(w2[0]-wBase[0]));
    if (0 == scalePerPixel)
        scalePerPixel = max(fabs(w1[2]-wBase[2]), fabs(w2[2]-wBase[2]));

    if (0==scalePerPixel) scalePerPixel=timePerPixel;
    if (0==timePerPixel) timePerPixel=scalePerPixel;

    // time/freqPerPixel is how much difference in time/freq there can be when moving one pixel away from the
    // pixel that represents the closest point in ref
}

template<typename T>
void swap( T& x, T& y) {
    x = x + y;
    y = x - y;
    x = x - y;
}


void Renderer::drawAxes( float T )
{
    last_axes_length = T;
    TIME_RENDERER TaskTimer tt("drawAxes(length = %g)", T);
    // Draw overlay borders, on top, below, to the right or to the left
    // default left bottom

    // 1 gray draw overlay
    // 2 clip entire sound to frustum
    // 3 decide upon scale
    // 4 draw axis
    unsigned screen_width = viewport_matrix[2];
    unsigned screen_height = viewport_matrix[3];

    float borderw = 12.5*1.1;
    float borderh = 12.5*1.1;

    float w = borderw/screen_width, h=borderh/screen_height;

    if (0) { // 1 gray draw overlay
        glPushMatrixContext push_model(GL_MODELVIEW);
        glPushMatrixContext push_proj(GL_PROJECTION);

        glLoadIdentity();
        gluOrtho2D( 0, 1, 0, 1 );

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);


        glColor4f( 1.0f, 1.0f, 1.0f, .4f );
        glBegin( GL_QUADS );
            glVertex2f( 0, 0 );
            glVertex2f( w, 0 );
            glVertex2f( w, 1 );
            glVertex2f( 0, 1 );

            glVertex2f( w, 0 );
            glVertex2f( 1-w, 0 );
            glVertex2f( 1-w, h );
            glVertex2f( w, h );

            glVertex2f( 1, 0 );
            glVertex2f( 1, 1 );
            glVertex2f( 1-w, 1 );
            glVertex2f( 1-w, 0 );

            glVertex2f( w, 1 );
            glVertex2f( w, 1-h );
            glVertex2f( 1-w, 1-h );
            glVertex2f( 1-w, 1 );
        glEnd();

        glEnable(GL_DEPTH_TEST);

        //glDisable(GL_BLEND);
    }

    // 2 clip entire sound to frustum
    clippedFrustum.clear();

    GLvector closest_i;
    {   //float T = collection->worker->source()->length();
        GLvector corner[4]=
        {
            GLvector( 0, 0, 0),
            GLvector( 0, 0, 1),
            GLvector( T, 0, 1),
            GLvector( T, 0, 0),
        };

        clippedFrustum = clipFrustum(corner, closest_i, w, h);
    }


    // 3 find inside
    GLvector inside;
    {
        for (unsigned i=0; i<clippedFrustum.size(); i++)
            inside = inside + clippedFrustum[i];

        // as clippedFrustum is a convex polygon, the mean position of its vertices will be inside
        inside = inside * (1.f/clippedFrustum.size());
    }


    // 4 render and decide upon scale
    GLvector x(1,0,0), z(0,0,1);

    glDepthMask(false);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Tfr::FreqAxis fa = collection->display_scale();
    // loop along all sides
    typedef tvector<4,GLfloat> GLvectorF;
    typedef tvector<2,GLfloat> GLvector2F;
    std::vector<GLvectorF> ticks;
    std::vector<GLvectorF> phatTicks;
    std::vector<GLvector2F> quad(4);

    for (unsigned i=0; i<clippedFrustum.size(); i++)
    {
        glColor4f(0,0,0,0.8);
        unsigned j=(i+1)%clippedFrustum.size();
        GLvector p1 = clippedFrustum[i]; // starting point of side
        GLvector p2 = clippedFrustum[j]; // end point of side
        GLvector v0 = p2-p1;

        // decide if this side is a t or f axis
        GLvector::T timePerPixel, scalePerPixel;
        computeUnitsPerPixel( inside, timePerPixel, scalePerPixel );

        bool taxis = fabsf(v0[0]*scalePerPixel) > fabsf(v0[2]*timePerPixel);


        // decide in which direction to traverse this edge
        GLvector::T timePerPixel1, scalePerPixel1, timePerPixel2, scalePerPixel2;
        computeUnitsPerPixel( p1, timePerPixel1, scalePerPixel1 );
        computeUnitsPerPixel( p2, timePerPixel2, scalePerPixel2 );

        double dscale = 0.001;
        double hzDelta1= fabs(fa.getFrequencyT( p1[2] + v0[2]*dscale ) - fa.getFrequencyT( p1[2] ));
        double hzDelta2 = fabs(fa.getFrequencyT( p2[2] - v0[2]*dscale ) - fa.getFrequencyT( p2[2] ));

        if ((taxis && timePerPixel1 > timePerPixel2) || (!taxis && hzDelta1 > hzDelta2))
        {
            GLvector flip = p1;
            p1 = p2;
            p2 = flip;
        }

        GLvector p = p1; // starting point
        GLvector v = p2-p1;

        if (!v[0] && !v[2]) // skip if |v| = 0
            continue;


        GLvector::T timePerPixel_closest, scalePerPixel_closest;
        computeUnitsPerPixel( closest_i, timePerPixel_closest, scalePerPixel_closest );

        if (draw_axis_at0==-1)
        {
            (taxis?p[2]:p[0]) = ((taxis?p[2]:p[0])==0) ? 1 : 0;
            (taxis?p1[2]:p1[0]) = ((taxis?p1[2]:p1[0])==0) ? 1 : 0;
        }

        // need initial f value
        GLvector pp = p;
        double f = fa.getFrequencyT( p[2] );

        if (((taxis && draw_t) || (!taxis && draw_hz)) &&
            (draw_axis_at0!=0?(taxis?p[2]==0:p[0]==0):true))
        for (double u=-1; true; )
        {
            GLvector::T timePerPixel, scalePerPixel;
            computeUnitsPerPixel( p, timePerPixel, scalePerPixel );
            double ppp=0.4;
            timePerPixel = timePerPixel * ppp + timePerPixel_closest * (1.0-ppp);
            scalePerPixel = scalePerPixel * ppp + scalePerPixel_closest * (1.0-ppp);

            double ST = timePerPixel * 750; // ST = time units per 750 pixels, 750 pixels is a fairly common window size
            double SF = scalePerPixel * 750;
            double drawScaleT = min(ST, 50*timePerPixel_closest*750);
            double drawScaleF = min(SF, 50*scalePerPixel_closest*750);

            double time_axis_density = 18;
            if (20.+2.*log10(timePerPixel) < 18.)
                time_axis_density = max(1., 20.+2.*log10(timePerPixel));

            double scale_axis_density = max(10., 22. - ceil(fabs(log10(f))));
            if (f == 0)
                scale_axis_density = 21;

            int st = floor(log10( ST / time_axis_density ));
            //int sf = floor(log10( SF / scale_axis_density ));

            double DT = pow(10.0, st);
            //double DF = pow(10, sf);
            double DF = min(0.2, SF / scale_axis_density);

            // compute index of next marker along t and f
            int tmultiple = 10, tsubmultiple = 5;

            if (st>0)
            {
                st = 0;
                if( 60 < ST ) DT = 10, st++, tmultiple = 6, tsubmultiple = 3;
                if( 60*10 < ST ) DT *= 6, st++, tmultiple = 10, tsubmultiple = 5;
                if( 60*10*6 < ST ) DT *= 10, st++, tmultiple = 6, tsubmultiple = 3;
                if( 60*10*6*24 < ST ) DT *= 6, st++, tmultiple = 24, tsubmultiple = 6;
                if( 60*10*6*24*5 < ST ) DT *= 24, st++, tmultiple = 5, tsubmultiple = 5;
            }

            int tupdatedetail = 1;
            DT /= tupdatedetail;
            int t = floor(p[0]/DT + .5); // t marker index along t
            double t2 = t*DT;
            if (t2 < p[0] && v[0] > 0)
                t++;
            if (t2 > p[0] && v[0] < 0)
                t--;
            p[0] = t*DT;

            //int tmarkanyways = (bool)(fabsf(5*DT*tupdatedetail) > (ST / time_axis_density) && ((unsigned)(p[0]/DT + 0.5)%(tsubmultiple*tupdatedetail)==0) && ((unsigned)(p[0]/DT +.5)%(tmultiple*tupdatedetail)!=0));
            int tmarkanyways = (bool)(fabsf(5*DT*tupdatedetail) > (ST / time_axis_density) && ((unsigned)(p[0]/DT + 0.5)%(tsubmultiple*tupdatedetail)==0));
            if (tmarkanyways)
                st--;

            // compute index of next marker along t and f
            double epsilon = 1.f/10;
            double hz1 = fa.getFrequencyT( p[2] - DF * epsilon );
            double hz2 = fa.getFrequencyT( p[2] + DF * epsilon );
            if (hz2-f < f-hz1)  hz1 = f;
            else                hz2 = f;
            double fc0 = (hz2 - hz1)/epsilon;
            int sf = floor(log10( fc0 ));
            double fc = powf(10, sf);
            int fmultiple = 10;

            int fupdatedetail = 1;
            fc /= fupdatedetail;
            int mif = floor(f / fc + .5); // f marker index along f
            double nf = mif * fc;
            if (!(f - fc*0.05 < nf && nf < f + fc*0.05))
                nf += v[2] > 0 ? fc : -fc;
            f = nf;
            p[2] = fa.getFrequencyScalarNotClampedT(f);

            double np1 = fa.getFrequencyScalarNotClampedT( f + fc);
            double np2 = fa.getFrequencyScalarNotClampedT( f - fc);
            int fmarkanyways = false;
            fmarkanyways |= 0.9*fabsf(np1 - p[2]) > DF && 0.9*fabsf(np2 - p[2]) > DF && ((unsigned)(f / fc + .5)%1==0);
            fmarkanyways |= 4.5*fabsf(np1 - p[2]) > DF && 4.5*fabsf(np2 - p[2]) > DF && ((unsigned)(f / fc + .5)%5==0);
            if (fmarkanyways)
                sf--;


            if (taxis && draw_cursor_marker)
            {
                float w = (cursor[0] - pp[0])/(p[0] - pp[0]);

                if (0 < w && w <= 1)
                    if (!tmarkanyways)
                        st--;

                if (fabsf(w) < tupdatedetail*tmultiple/2)
                    tmarkanyways = -1;

                if (0 < w && w <= 1)
                {
                    DT /= 10;
                    t = floor(cursor[0]/DT + 0.5); // t marker index along t

                    p = p1 + v*((cursor[0] - p1[0])/v[0]);
                    p[0] = cursor[0]; // exact float value so that "cursor[0] - pp[0]" == 0

                    if (!tmarkanyways)
                        st--;

                    tmarkanyways = 2;
                }
            }
            else if(draw_cursor_marker)
            {
                float w = (cursor[2] - pp[2])/(p[2] - pp[2]);

                if (0 < w && w <= 1)
                    if (!fmarkanyways)
                        sf--;

                if (fabsf(w) < fupdatedetail*fmultiple/2)
                    fmarkanyways = -1;

                if (0 < w && w <= 1)
                {
                    f = fa.getFrequencyT( cursor[2] );
                    fc /= 10;
                    mif = floor(f / fc + .5); // f marker index along f
                    f = mif * fc;

                    p = p1 + v*((cursor[2] - p1[2])/v[2]);
                    p[2] = cursor[2]; // exact float value so that "cursor[2] - pp[2]" == 0

                    fmarkanyways = 2;
                }
            }


            // find next intersection along v
            double nu;
            int c1 = taxis ? 0 : 2;
            int c2 = !taxis ? 0 : 2;
            nu = (p[c1] - p1[c1])/v[c1];

            // if valid intersection
            if ( nu > u && nu<=1 ) { u = nu; }
            else break;

            // compute intersection
            p[c2] = p1[c2] + v[c2]*u;


            GLvector np = p;
            nf = f;
            int nt = t;

            if (taxis)
            {
                if (v[0] > 0) nt++;
                if (v[0] < 0) nt--;
                np[0] = nt*DT;
            }
            else
            {
                if (v[2] > 0) nf+=fc;
                if (v[2] < 0) nf-=fc;
                nf = floor(nf/fc + .5)*fc;
                np[2] = fa.getFrequencyScalarNotClampedT(nf);
            }

            // draw marker
            if (taxis) {
                if (0 == t%tupdatedetail || tmarkanyways==2)
                {
                    float size = 1+ (0 == (t%(tupdatedetail*tmultiple)));
                    if (tmarkanyways)
                        size = 2;
                    if (-1 == tmarkanyways)
                        size = 1;

                    float sign = (v0^z)%(v0^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*SF*.003f*sign;

                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2], 1.f));
                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2]-o, 1.f));

                    if (size>1) {
                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0], 0, p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*drawScaleT,0.00013f*drawScaleF,1.f);
                        float angle = atan2(v0[2]/SF, v0[0]/ST) * (180*M_1_PI);
                        glRotatef(angle,0,0,1);
                        char a[100];
                        char b[100];
                        sprintf(b,"%%d:%%0%d.%df", 2+(st<-1?-st:0), st<0?-1-st:0);
                        int minutes = (int)(t*DT/60);
                        sprintf(a, b, minutes,t*DT-60*minutes);
                        float w=0;
                        float letter_spacing=15;

                        for (char*c=a;*c!=0; c++) {
                            if (c!=a)
                                w+=letter_spacing;
                            w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                        }

                        if (!left_handed_axes)
                            glScalef(-1,1,1);
                        glTranslatef(0,70.f,0);
                        if (sign<0)
                            glRotatef(180,0,0,1);
                        glTranslatef(-.5f*w,-50.f,0);
                        glColor4f(1,1,1,0.5);
                        float z = 10;
                        float q = 20;
                        glEnableClientState(GL_VERTEX_ARRAY);
                        quad[0] = GLvector2F(0 - z, 0 - q);
                        quad[1] = GLvector2F(w + z, 0 - q);
                        quad[2] = GLvector2F(0 - z, 100 + q);
                        quad[3] = GLvector2F(w + z, 100 + q);
                        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
                        glDisableClientState(GL_VERTEX_ARRAY);
                        glColor4f(0,0,0,0.8);
                        for (char*c=a;*c!=0; c++) {
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                            glTranslatef(letter_spacing,0,0);
                        }
                    }
                }
            } else {
                if (0 == ((unsigned)floor(f/fc + .5))%fupdatedetail || fmarkanyways==2)
                {
                    int size = 1;
                    if (0 == ((unsigned)floor(f/fc + .5))%(fupdatedetail*fmultiple))
                        size = 2;
                    if (fmarkanyways)
                        size = 2;
                    if (-1 == fmarkanyways)
                        size = 1;


                    float sign = (v0^x)%(v0^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*ST*.003f*sign;

                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2], 1.f));
                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0]-o, 0.f, p[2], 1.f));


                    if (size>1)
                    {
                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0],0,p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*drawScaleT,0.00013f*drawScaleF,1.f);
                        float angle = atan2(v0[2]/SF, v0[0]/ST) * (180*M_1_PI);
                        glRotatef(angle,0,0,1);
                        char a[100];
                        char b[100];
                        sprintf(b,"%%.%df", sf<0?-1-sf:0);
                        sprintf(a, b, f);
                        //sprintf(a,"%g", f);
                        unsigned w=0;
                        float letter_spacing=5;

                        for (char*c=a;*c!=0; c++)
                        {
                            if (c!=a)
                                w+=letter_spacing;
                            w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                        }
                        if (!left_handed_axes)
                            glScalef(-1,1,1);
                        glTranslatef(0,70.f,0);
                        if (sign<0)
                            glRotatef(180,0,0,1);
                        glTranslatef(-.5f*w,-50.f,0);
                        glColor4f(1,1,1,0.5);
                        float z = 10;
                        float q = 20;
                        glEnableClientState(GL_VERTEX_ARRAY);
                        quad[0] = GLvector2F(0 - z, 0 - q);
                        quad[1] = GLvector2F(w + z, 0 - q);
                        quad[2] = GLvector2F(0 - z, 100 + q);
                        quad[3] = GLvector2F(w + z, 100 + q);
                        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
                        glDisableClientState(GL_VERTEX_ARRAY);
                        glColor4f(0,0,0,0.8);
                        for (char*c=a;*c!=0; c++)
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                    }
                }
            }

            pp = p;
            p = np; // q.e.d.
            f = nf;
            t = nt;
        }

        if (!taxis && draw_piano && (draw_axis_at0?p[0]==0:true))
        {
            GLvector::T timePerPixel, scalePerPixel;
            computeUnitsPerPixel( p + v*0.5, timePerPixel, scalePerPixel );

            double ST = timePerPixel * 750;
            // double SF = scalePerPixel * 750;

            // from http://en.wikipedia.org/wiki/Piano_key_frequencies
            // F(n) = 440 * pow(pow(2, 1/12),n-49)
            // log(F(n)/440) = log(pow(2, 1/12),n-49)
            // log(F(n)/440) = log(pow(2, 1/12))*log(n-49)
            // log(F(n)/440)/log(pow(2, 1/12)) = log(n-49)
            // n = exp(log(F(n)/440)/log(pow(2, 1/12))) + 49

            unsigned F1 = fa.getFrequency( (float)p1[2] );
            unsigned F2 = fa.getFrequency( (float)(p1+v)[2] );
            if (F2<F1) { unsigned swap = F2; F2=F1; F1=swap; }
            if (!(F1>fa.min_hz)) F1=fa.min_hz;
            if (!(F2<fa.max_hz())) F2=fa.max_hz();
            float tva12 = powf(2.f, 1.f/12);


            if (0 == F1)
                F1 = 1;
            int startTone = log(F1/440.f)/log(tva12) + 45;
            int endTone = ceil(log(F2/440.f)/log(tva12)) + 45;
            float sign = (v^x)%(v^( clippedFrustum[i] - inside))>0 ? 1.f : -1.f;
            if (!left_handed_axes)
                sign *= -1;

            for( int tone = startTone; tone<=endTone; tone++)
            {
                float ff = fa.getFrequencyScalar(440 * pow(tva12,tone-45));
                float ffN = fa.getFrequencyScalar(440 * pow(tva12,tone-44));
                float ffP = fa.getFrequencyScalar(440 * pow(tva12,tone-46));

                int toneTest = tone;
                while(toneTest<0) toneTest+=12;

                bool blackKey = false;
                switch(toneTest%12) { case 1: case 3: case 6: case 8: case 10: blackKey = true; }
                bool blackKeyP = false;
                switch((toneTest+11)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyP = true; }
                bool blackKeyN = false;
                switch((toneTest+1)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyN = true; }
                glLineWidth(1);
                float wN = ffN-ff, wP = ff-ffP;
                if (blackKey)
                    wN *= .5, wP *= .5;
                else {
                    if (!blackKeyN)
                        wN *= .5;
                    if (!blackKeyP)
                        wP *= .5;
                }

                float u = (ff - p1[2])/v[2];
                float un = (ff+wN - p1[2])/v[2];
                float up = (ff-wP - p1[2])/v[2];
                GLvector pt = p1+v*u;
                GLvector pn = p1+v*un;
                GLvector pp = p1+v*up;

                glPushMatrixContext push_model( GL_MODELVIEW );

                float xscale = 0.016000f;
                float blackw = 0.4f;

                if (sign>0)
                    glTranslatef( xscale*ST, 0.f, 0.f );

                tvector<4,GLfloat> keyColor(0,0,0, 0.7f * blackKey);
                if (draw_cursor_marker)
                {
                    float w = (cursor[2] - ff)/(ffN - ff);
                    w = fabsf(w/1.6f);
                    if (w < 1)
                    {
                        keyColor[1] = (1-w)*(1-w);
                        if (blackKey)
                            keyColor[3] = keyColor[3]*w + .9f*(1-w);
                        else
                            keyColor[3] = keyColor[1] * .7f;
                    }
                }

                if (keyColor[3] != 0)
                {
                    glColor4fv(keyColor.v);
                    if (blackKey)
                    {
                        glBegin(GL_TRIANGLE_STRIP);
                            glVertex3f(pp[0] - xscale*ST*(1.f), 0, pp[2]);
                            glVertex3f(pp[0] - xscale*ST*(blackw), 0, pp[2]);
                            glVertex3f(pn[0] - xscale*ST*(1.f), 0, pn[2]);
                            glVertex3f(pn[0] - xscale*ST*(blackw), 0, pn[2]);
                        glEnd();
                    }
                    else
                    {
                        glBegin(GL_TRIANGLE_FAN);
                            if (blackKeyP)
                            {
                                glVertex3dv((pp*0.5 + pt*0.5 - GLvector(xscale*ST*(blackw), 0, 0)).v);
                                glVertex3f(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                            }
                            glVertex3f(pp[0] - xscale*ST*(0.f), 0, pp[2]);
                            glVertex3f(pn[0] - xscale*ST*(0.f), 0, pn[2]);
                            glVertex3f(pn[0] - xscale*ST*(blackKeyN ? blackw : 1.f), 0, pn[2]);
                            if (blackKeyN)
                            {
                                glVertex3dv((pn*0.5 + pt*0.5 - GLvector(xscale*ST*(blackw), 0, 0)).v);
                                glVertex3dv((pn*0.5 + pt*0.5 - GLvector(xscale*ST*(1.f), 0, 0)).v);
                            }
                            if (blackKeyP)
                                glVertex3dv((pp*0.5 + pt*0.5 - GLvector(xscale*ST*(1.f), 0, 0)).v);
                            else
                                glVertex3f(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                        glEnd();
                    }
                }

                // outline
                glColor4f(0,0,0,0.8);
                    glBegin(GL_LINES );
                        glVertex3f(pn[0] - xscale*ST, 0, pn[2]);
                        glVertex3f(pp[0] - xscale*ST, 0, pp[2]);
                    glEnd();
                    glBegin(GL_LINE_STRIP);
                        glVertex3f(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                        glVertex3f(pp[0] - xscale*ST*(blackKey ? blackw : 0.f), 0, pp[2]);
                        glVertex3f(pn[0] - xscale*ST*(blackKey ? blackw : 0.f), 0, pn[2]);
                        glVertex3f(pn[0] - xscale*ST*(blackKeyN ? blackw : 1.f), 0, pn[2]);
                    glEnd();

                glColor4f(0,0,0,0.8);

                if (tone%12 == 0)
                {
                    glLineWidth(1);

                    glPushMatrixContext push_model( GL_MODELVIEW );
                    glTranslatef(pp[0], 0, pp[2]);
                    glRotatef(90,1,0,0);

                    //glScalef(0.00014f*ST,0.00014f*SF,1.f);
                    glScalef(0.005 * xscale*ST,0.35 * xscale*(pn[2]-pp[2]),1.f);

                    char a[100];
                    sprintf(a,"C%d", tone/12+1);
                    float w=10;
                    for (char*c=a;*c!=0; c++)
                        w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                    if (!left_handed_axes)
                        glScalef(-1,1,1);
                    glTranslatef(-w,0,0);
                    glColor4f(1,1,1,0.5);
                    float z = 10;
                    float q = 20;
                    glBegin(GL_TRIANGLE_STRIP);
                        glVertex2f(0 - z, 0 - q);
                        glVertex2f(w + z, 0 - q);
                        glVertex2f(0 - z, 100 + q);
                        glVertex2f(w + z, 100 + q);
                    glEnd();
                    glColor4f(0,0,0,0.9);
                    for (char*c=a;*c!=0; c++)
                        glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                }
            }
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    if (!phatTicks.empty())
    {
        glLineWidth(2);
        glVertexPointer(4, GL_FLOAT, 0, &phatTicks[0]);
        glDrawArrays(GL_LINES, 0, phatTicks.size());
        glLineWidth(1);
    }
    if (!ticks.empty())
    {
        glVertexPointer(4, GL_FLOAT, 0, &ticks[0]);
        glDrawArrays(GL_LINES, 0, ticks.size());
    }
    glDisableClientState(GL_VERTEX_ARRAY);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(true);
}

template<typename T> void glVertex3v( const T* );

template<> void glVertex3v( const GLdouble* t ) {    glVertex3dv(t); }
template<>  void glVertex3v( const GLfloat* t )  {    glVertex3fv(t); }

void Renderer::
        drawFrustum()
{
    if (clippedFrustum.empty())
        return;

    GLvector closest = clippedFrustum.front();
    for ( std::vector<GLvector>::const_iterator i = clippedFrustum.begin();
            i!=clippedFrustum.end();
            i++)
    {
        if ((closest-camera).dot() > (*i - camera).dot())
            closest = *i;
    }


    glPushAttribContext ac;

    glDisable(GL_DEPTH_TEST);

    glPushMatrixContext mc(GL_MODELVIEW);

    glEnable(GL_BLEND);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, &clippedFrustum[0]);


    // dark inside
    float darkness = 0.2f; // 0 = not dark, 1 = very dark
    glColor4f( darkness, darkness, darkness, 1 );
    glBlendEquation( GL_FUNC_REVERSE_SUBTRACT );
    glBlendFunc( GL_ONE_MINUS_DST_COLOR, GL_ONE );
    glDrawArrays( GL_TRIANGLE_FAN, 0, clippedFrustum.size() );
    glBlendEquation( GL_FUNC_ADD );


    // black border
    glColor4f( 0, 0, 0, 0.5 );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth( 0.5 );
    glDrawArrays(GL_LINE_LOOP, 0, clippedFrustum.size());


    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_BLEND);
}

} // namespace Heightmap
