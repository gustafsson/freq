#include <gl.h>

#include "heightmap/renderer.h"

#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif

#include <float.h>
#include <GlException.h>
#include <computationkernel.h>
#include <glPushContext.h>

#include <boost/foreach.hpp>

#ifdef _MSC_VER
#include "msc_stdc.h"
#endif

//#define TIME_RENDERER
#define TIME_RENDERER if(0)

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

namespace Heightmap {


using namespace std;

Renderer::Renderer( Collection* collection )
:   collection(collection),
    draw_piano(false),
    draw_hz(true),
    draw_t(true),
    draw_cursor_marker(false),
    camera(0,0,0),
    draw_height_lines(false),
    color_mode( ColorMode_Rainbow ),
    fixed_color( 1,0,0,1 ),
    y_scale( 1 ),
    last_ysize( 1 ),
    drawn_blocks(0),
    left_handed_axes(true),
    _mesh_index_buffer(0),
    _mesh_width(0),
    _mesh_height(0),
    _mesh_fraction_width(1),
    _mesh_fraction_height(1),
    _initialized(NotInitialized),
    _draw_flat(false),
    _redundancy(3.0), // 1 means every pixel gets its own vertex, 10 means every 10th pixel gets its own vertex, default=2
    _invalid_frustum(true)

{
    memset(modelview_matrix, 0, sizeof(modelview_matrix));
    memset(projection_matrix, 0, sizeof(projection_matrix));
    memset(viewport_matrix, 0, sizeof(viewport_matrix));
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
    // create index buffer
    if (_mesh_index_buffer)
        glDeleteBuffersARB(1, &_mesh_index_buffer);

    _mesh_width = w;
    _mesh_height = h;

    _vbo_size = ((w*2)+4)*(h-1);
    glGenBuffersARB(1, &_mesh_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, _vbo_size*sizeof(GLuint), 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!indices) {
        return;
    }

    for(unsigned y=0; y<h-1; y++) {
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

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void Renderer::createMeshPositionVBO(unsigned w, unsigned h)
{
    _mesh_position.reset( new Vbo( w*h*4*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW ));

    glBindBuffer(_mesh_position->vbo_type(), *_mesh_position);
    float *pos = (float *) glMapBuffer(_mesh_position->vbo_type(), GL_WRITE_ONLY);
    if (!pos) {
        return;
    }

    for(unsigned y=0; y<h-1; y++) {
        for(unsigned x=0; x<w; x++) {
            float u = x / (float) (w-1);
            float v = y / (float) (h-1);
            *pos++ = u;
            *pos++ = 0.0f;
            *pos++ = v;
            *pos++ = 1.0f;
        }
    }

    for(unsigned x=0; x<w; x++) {
        float u = x / (float) (w-1);
        float v = 1;
        *pos++ = u;
        *pos++ = 0.0f;
        *pos++ = v;
        *pos++ = 1.0f;
    }

    glUnmapBuffer(_mesh_position->vbo_type());
    glBindBuffer(_mesh_position->vbo_type(), 0);
}

typedef tvector<4,GLdouble> GLvector4;
typedef tmatrix<4,GLdouble> GLmatrix;

// static GLvector4 to4(const GLvector& a) { return GLvector4(a[0], a[1], a[2], 1);}
// static GLvector to3(const GLvector4& a) { return GLvector(a[0], a[1], a[2]);}

GLvector gluProject(GLvector obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r) {
    GLdouble win0=0, win1=0, win2=0;
    bool s = (GLU_TRUE == ::gluProject(obj[0], obj[1], obj[2], model, proj, view, &win0, &win1, &win2));
    if(r) *r=s;
    return GLvector(win0, win1, win2);
}

GLvector gluUnProject(GLvector win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r) {
    GLdouble obj0=0, obj1=0, obj2=0;
    bool s = (GLU_TRUE == ::gluUnProject(win[0], win[1], win[2], model, proj, view, &obj0, &obj1, &obj2));
    if(r) *r=s;
    return GLvector(obj0, obj1, obj2);
}

/*
template<typename f>
GLvector gluProject2(tvector<3,f> obj, bool *r) {
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);
    GLvector4 eye = applyProjectionMatrix(applyModelMatrix(to4(obj)));
    eye[0]/=eye[3];
    eye[1]/=eye[3];
    eye[2]/=eye[3];

    GLvector screen(view[0] + (eye[0]+1)*view[2]/2, view[1] + (eye[1]+1)*view[3]/2, eye[2]);
    float dummy=43*23;
    return screen;
}*/


/* static bool validWindowPos(GLvector win) {
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);

    return win[0]>view[0] && win[1]>view[1]
            && win[0]<view[0]+view[2]
            && win[1]<view[1]+view[3]
            && win[2]>=0.1 && win[2]<=100;
}*/

/*static GLvector4 applyModelMatrix(GLvector4 obj) {
    GLdouble m[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);

    GLvector4 eye = GLmatrix(m) * obj;
    return eye;
}*/

/*static GLvector4 applyProjectionMatrix(GLvector4 eye) {
    GLdouble p[16];
    glGetDoublev(GL_PROJECTION_MATRIX, p);

    GLvector4 clip = GLmatrix(p) * eye;
    return clip;
}*/

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

    // initialize necessary OpenGL extensions
    GlException_CHECK_ERROR();

#ifdef USE_CUDA
    int cudaDevices=0;
    CudaException_SAFE_CALL( cudaGetDeviceCount( &cudaDevices) );
    if (0 == cudaDevices ) {
        fprintf(stderr, "ERROR: Couldn't find any \"cuda capable\" device.");
        fflush(stderr);
        throw std::runtime_error("Could not find any \"cuda capable\" device");
    }
#endif

#ifndef __APPLE__
    if (0 != glewInit() ) {
        fprintf(stderr, "ERROR: Couldn't initialize \"glew\".");
        fflush(stderr);
        throw std::runtime_error("Could not initialize glew");
    }

    if (! glewIsSupported("GL_VERSION_2_0" )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions (GL_VERSION_2_0) missing.");
        fflush(stderr);
        throw std::runtime_error("Sonic AWE requires a graphics card that supports OpenGL 2.0.\n\nNo such graphics card was found, Sonic AWE can not start.");
    }

    if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object GL_ARB_texture_float" )) {
        fprintf(stderr, "Error: failed to get minimal extensions\n");
        fprintf(stderr, "Sonic AWE requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        fprintf(stderr, "  GL_ARB_texture_float\n");
        fflush(stderr);
        throw std::runtime_error(
                "Sonic AWE requires a graphics card that supports OpenGL 1.5 and the following OpenGL features\n"
                "  GL_ARB_vertex_buffer_object\n"
                "  GL_ARB_pixel_buffer_object\n"
                "  GL_ARB_texture_float.\n\n"
                "No such graphics card was found, Sonic AWE can not start.");
    }
#endif

    // load shader
    _shader_prog = loadGLSLProgram(":/shaders/heightmap.vert", ":/shaders/heightmap.frag");

    //setSize( collection->samples_per_block(), collection->scales_per_block() );
    setSize( collection->samples_per_block()/4, collection->scales_per_block() );

    //setSize(2,2);

    createColorTexture(16); // These will be linearly interpolated when rendering, so a high resolution texture is not needed

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


tvector<4,float> mix(tvector<4,float> a, tvector<4,float> b, float f)
{
    return a*(1-f)+b;
}

tvector<4,float> getWavelengthColorCompute( float wavelengthScalar ) {
    tvector<4,float> spectrum[7];
        /* white background */
    spectrum[0] = tvector<4,float>( 1, 0, 0, 0 ),
    spectrum[1] = tvector<4,float>( 0, 0, 1, 0 ),
    spectrum[2] = tvector<4,float>( 0, 1, 1, 0 ),
    spectrum[3] = tvector<4,float>( 0, 1, 0, 0 ),
    spectrum[4] = tvector<4,float>( 1, 1, 0, 0 ),
    spectrum[5] = tvector<4,float>( 1, 0, 1, 0 ),
    spectrum[6] = tvector<4,float>( 1, 0, 0, 0 );
        /* black background
        { 0, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }}; */

    int count = 6;//sizeof(spectrum)/sizeof(spectrum[0])-1;
    float f = float(count)*wavelengthScalar;
    int i1 = int(floor(max(0.f, min(f-1.f, float(count)))));
    int i2 = int(floor(min(f, float(count))));
    int i3 = int(floor(min(f+1.f, float(count))));
    int i4 = int(floor(min(f+2.f, float(count))));
    float t = (f-float(i2))*0.5;
    float s = 0.5 + t;

    tvector<4,float> rgb = mix(spectrum[i1], spectrum[i3], s) + mix(spectrum[i2], spectrum[i4], t);
    return rgb*0.5;
}

void Renderer::createColorTexture(unsigned N) {
    std::vector<tvector<4,float> > texture(N);
    for (unsigned i=0; i<N; ++i) {
        texture[i] = getWavelengthColorCompute( i/(float)(N-1) );
    }
    colorTexture.reset( new GlTexture(N,1, GL_RGBA, GL_RGBA, GL_FLOAT, &texture[0]));
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

    while(true)
    {
        LevelOfDetal lod = testLod(ref);

        Position a,b;
        ref.getArea(a,b);

        switch(lod)
        {
        case Lod_NeedBetterF:
            if ((a.scale+b.scale)/2 > p.scale)
                ref = ref.bottom();
            else
                ref = ref.top();
            break;

        case Lod_NeedBetterT:
            if ((a.time+b.time)/2 > p.time)
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

    setSize( collection->samples_per_block()/_mesh_fraction_width, collection->scales_per_block()/_mesh_fraction_height );

    _invalid_frustum = true;

    if (.001 > scaley)
//        setSize(2,2),
        scaley = 0.001,
        _draw_flat = true;
    else
        _draw_flat = false;

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

    glUseProgram(_shader_prog);

    // TODO check if this takes any time
    {   // Set default uniform variables parameters for the vertex and pixel shader
        GLuint uniVertText0, uniVertText1, uniVertText2, uniColorMode, uniFixedColor, uniHeightLines, uniYScale, uniScaleTex, uniOffsTex;

        uniVertText0 = glGetUniformLocation(_shader_prog, "tex");
        glUniform1i(uniVertText0, 0); // GL_TEXTURE0

        uniVertText1 = glGetUniformLocation(_shader_prog, "tex_nearest");
        glUniform1i(uniVertText1, 1); // GL_TEXTURE1

        uniVertText2 = glGetUniformLocation(_shader_prog, "tex_color");
        glUniform1i(uniVertText2, 2); // GL_TEXTURE2

        uniColorMode = glGetUniformLocation(_shader_prog, "colorMode");
        glUniform1i(uniColorMode, (int)color_mode);

        uniFixedColor = glGetUniformLocation(_shader_prog, "fixedColor");
        glUniform4f(uniFixedColor, fixed_color[0], fixed_color[1], fixed_color[2], fixed_color[3]);

        uniHeightLines = glGetUniformLocation(_shader_prog, "heightLines");
        glUniform1i(uniHeightLines, draw_height_lines && !_draw_flat);

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

    glActiveTexture(GL_TEXTURE2);
    colorTexture->bindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    if (!_draw_flat)
    {
        glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);
    }

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

    Position a, b;
    ref.getArea( a, b );
    glPushMatrixContext mc( GL_MODELVIEW );

    glTranslatef(a.time, 0, a.scale);
    glScalef(b.time-a.time, 1, b.scale-a.scale);

    pBlock block = collection->getBlock( ref );
    if (0!=block.get()) {
        if (0 /* direct rendering */ )
            ;//block->glblock->draw_directMode();
        else if (1 /* vbo */ )
        {
            if (_draw_flat)
                block->glblock->draw_flat();
            else
                block->glblock->draw( _vbo_size );
        }

    } else if ( 0 == "render red warning cross") {
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

        if (d%v<0) allNeg=false;
        if (d%v>0) allPos=false;
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
            n = n.Normalize();
            r = target + n*distanceToPlane( target, l[0], n );
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
        projectionNormal = (gluUnProject( GLvector( view[0] + view[2]/2, view[1] + view[3]/2, z1) ) - projectionPlane).Normalize();

        rightPlane = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z0) );
        GLvector rightZ = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2, z1) );
        GLvector rightY = gluUnProject( GLvector( view[0] + (1-w)*view[2], view[1] + view[3]/2+1, z0) );
        rightZ = rightZ - rightPlane;
        rightY = rightY - rightPlane;
        rightNormal = ((rightY)^(rightZ)).Normalize();

        leftPlane = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z0) );
        GLvector leftZ = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2, z1) );
        GLvector leftY = gluUnProject( GLvector( view[0]+w*view[2], view[1] + view[3]/2+1, z0) );
        leftNormal = ((leftZ-leftPlane)^(leftY-leftPlane)).Normalize();

        topPlane = gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z0) );
        GLvector topZ = gluUnProject( GLvector( view[0] + view[2]/2, view[1] + (1-h)*view[3], z1) );
        GLvector topX = gluUnProject( GLvector( view[0] + view[2]/2+1, view[1] + (1-h)*view[3], z0) );
        topNormal = ((topZ-topPlane)^(topX-topPlane)).Normalize();

        bottomPlane = gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z0) );
        GLvector bottomZ = gluUnProject( GLvector( view[0] + view[2]/2, view[1]+h*view[3], z1) );
        GLvector bottomX = gluUnProject( GLvector( view[0] + view[2]/2+1, view[1]+h*view[3], z0) );
        bottomNormal = ((bottomX-bottomPlane)^(bottomZ-bottomPlane)).Normalize();

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
        l.push_back( corner[i] );
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
    Position p[2];
    ref.getArea( p[0], p[1] );

    GLvector corner[4]=
    {
        GLvector( p[0].time, 0, p[0].scale),
        GLvector( p[0].time, 0, p[1].scale),
        GLvector( p[1].time, 0, p[1].scale),
        GLvector( p[1].time, 0, p[0].scale)
    };

    GLvector closest_i;
    std::vector<GLvector> clippedCorners = clipFrustum(corner, closest_i);
    if (0) if (-10==ref.log2_samples_size[0] && -8==ref.log2_samples_size[1])
    {
        printl("Clipped corners",clippedCorners);
        printf("closest_i %g\t%g\t%g\n", closest_i[0], closest_i[1], closest_i[2]);
    }
    if (0==clippedCorners.size())
        return false;

    GLvector::T
            timePerPixel = 0,
            freqPerPixel = 0;
    if (!computeUnitsPerPixel( closest_i, timePerPixel, freqPerPixel ))
        return false;

    // time/scalepixels is approximately the number of pixels in ref along the time/scale axis
    timePixels = (p[1].time - p[0].time)/timePerPixel;
    scalePixels = (p[1].scale - p[0].scale)/freqPerPixel;

    return true;
}


bool Renderer::
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

    return true;
}

template<typename T>
void swap( T& x, T& y) {
    x = x + y;
    y = x - y;
    x = x - y;
}


void Renderer::drawAxes( float T )
{
    TaskTimer tt("drawAxes(length = %g)", T);
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

    if (!left_handed_axes)
    {
        swap( h, w );
    }

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

    {   //float T = collection->worker->source()->length();
        GLvector closest_i;
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
    for (unsigned i=0; i<clippedFrustum.size(); i++)
    {
        glColor4f(0,0,0,0.8);
        unsigned j=(i+1)%clippedFrustum.size();
        GLvector p = clippedFrustum[i]; // starting point of side
        GLvector v = clippedFrustum[j]-p; // vector pointing from p to the next vertex

        if (!v[0] && !v[2]) // skip if |v| = 0
            continue;

        // decide if this side is a t or f axis
        GLvector::T timePerPixel, scalePerPixel;
        computeUnitsPerPixel( inside, timePerPixel, scalePerPixel );

        bool taxis = fabsf(v[0]*scalePerPixel) > fabsf(v[2]*timePerPixel);

        double f = fa.getFrequencyT( p[2] );

        if ((taxis && draw_t) || (!taxis && draw_hz))
        for (double u=-1; true; )
        {
            GLvector::T timePerPixel, scalePerPixel;
            computeUnitsPerPixel( p, timePerPixel, scalePerPixel );

            double ST = timePerPixel * 750;
            double SF = scalePerPixel * 750;

            float time_axis_density = 18;
            if (20.f+2.f*log10(timePerPixel) < 18.f)
                time_axis_density = std::max(1., 20.f+2.f*log10(timePerPixel));

            float scale_axis_density = 3;

            int st = floor(log10( ST / time_axis_density ));
            int sf = floor(log10( SF / scale_axis_density ));

            double DT = powf(10, st);
            double DF = powf(10, sf);

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

            int tmarkanyways = (bool)(fabsf(5*DT) > (ST / time_axis_density) && ((unsigned)(p[0]/DT)%tsubmultiple==0) && ((unsigned)(p[0]/DT)%tmultiple!=0));
            if (tmarkanyways)
                st--;

            int tupdatedetail = 2;
            DT /= tupdatedetail;
            int t = p[0]/DT; // t marker index along t
            if (v[0] > 0 && p[0] > t*DT) t++;


            // compute index of next marker along t and f
            double epsilon = 1.f/50;
            double hz1 = fa.getFrequencyT( p[2] - DF/2 * epsilon );
            double hz2 = fa.getFrequencyT( p[2] + DF/2 * epsilon );
            double fc0 = (hz2 - hz1)/epsilon;
            sf = floor(log10( fc0 ));
            double fc = powf(10, sf);
            int fmultiple = 10;
            double np1 = fa.getFrequencyScalarNotClampedT( f + fc);
            double np2 = fa.getFrequencyScalarNotClampedT( f - fc);
            int fmarkanyways = (bool)(fabsf(5*DF) > (SF / scale_axis_density) && ((unsigned)(f / fc + .5)%5==0) && ((unsigned)(f / fc + .5)%fmultiple!=0));
            fmarkanyways |= 7*fabsf(np1 - p[2]) > (SF / scale_axis_density) && 7*fabsf(np2 - p[2]) > (SF / scale_axis_density);
            if (fmarkanyways)
                sf--;

            int fupdatedetail = 2;
            fc /= fupdatedetail;
            int mif = floor(f / fc + .5); // f marker index along f
            f = mif * fc;

            if (v[2] > 0 && p[2]>fa.getFrequencyScalarNotClampedT(f)) f+=fc;


            // find next intersection along v
            double nu;
            if (taxis)  nu = (p[0] - clippedFrustum[i][0])/v[0];
            else        nu = (p[2] - clippedFrustum[i][2])/v[2];

            // if valid intersection
            if ( nu > u && nu<=1 ) { u = nu; }
            else break;

            // compute intersection
            p = clippedFrustum[i] + v*u;


            GLvector np = p;
            float nf = f;
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

            if (taxis && draw_cursor_marker)
            {
                float w = (cursor[0] - p[0])/(np[0] - p[0]);

                if (0 <= w && w < 1)
                    if (!tmarkanyways)
                        st--;

                if (fabsf(w) < tupdatedetail*tmultiple/2)
                    tmarkanyways = -1;

                if (0 <= w && w < 1)
                {
                    p[0] = cursor[0];
                    DT /= 10;
                    t = cursor[0]/DT; // t marker index along t

                    p = clippedFrustum[i] + v*((cursor[0] - clippedFrustum[i][0])/v[0]);

                    if (!tmarkanyways)
                        st--;

                    tmarkanyways = 2;
                }
            }
            else if(draw_cursor_marker)
            {
                float w = (cursor[2] - p[2])/(np[2] - p[2]);

                if (0 <= w && w < 1)
                    if (!fmarkanyways)
                        sf--;

                if (fabsf(w) < fupdatedetail*fmultiple/2)
                    fmarkanyways = -1;

                if (0 <= w && w < 1)
                {
                    f = fa.getFrequencyT( cursor[2] );
                    fc /= 10;
                    mif = floor(f / fc + .5); // f marker index along f
                    f = mif * fc;

                    p = clippedFrustum[i] + v*((cursor[2] - clippedFrustum[i][2])/v[2]);

                    fmarkanyways = 2;
                }
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

                    glLineWidth(size);

                    float sign = (v^z)%(v^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*SF*.003f*sign;

                    glBegin(GL_LINES);
                        glVertex3f( p[0], 0, p[2] );
                        glVertex3f( p[0], 0, p[2] - o);
                    glEnd();

                    if (size>1) {
                        glLineWidth(1);

                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0], 0, p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*ST,0.00013f*SF,1.f);
                        float angle = atan2(v[2]/SF, v[0]/ST) * (180*M_1_PI);
                        glRotatef(angle,0,0,1);
                        char a[100];
                        char b[100];
                        sprintf(b,"%%d:%%02.%df", st<0?-1-st:0);
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
                        glBegin(GL_TRIANGLE_STRIP);
                        glVertex2f(0 - z, 0 - q);
                        glVertex2f(w + z, 0 - q);
                        glVertex2f(0 - z, 100 + q);
                        glVertex2f(w + z, 100 + q);
                        glEnd();
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
                    float size = 1;
                    if (0 == ((unsigned)floor(f/fc + .5))%(fupdatedetail*fmultiple))
                        size = 2;
                    if (fmarkanyways)
                        size = 2;
                    if (-1 == fmarkanyways)
                        size = 1;


                    glLineWidth(size);

                    float sign = (v^x)%(v^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*ST*.003f*sign;
                    if (!left_handed_axes)
                        sign *= -1;
                    glBegin(GL_LINES);
                        glVertex3f( p[0], 0, p[2] );
                        glVertex3f( p[0] - o, 0, p[2] );
                    glEnd();


                    if (size>1)
                    {
                        glLineWidth(1);

                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0],0,p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*ST,0.00013f*SF,1.f);
                        float angle = atan2(v[2]/SF, v[0]/ST) * (180*M_1_PI);
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
                        glBegin(GL_TRIANGLE_STRIP);
                        glVertex2f(0 - z, 0 - q);
                        glVertex2f(w + z, 0 - q);
                        glVertex2f(0 - z, 100 + q);
                        glVertex2f(w + z, 100 + q);
                        glEnd();
                        glColor4f(0,0,0,0.8);
                        for (char*c=a;*c!=0; c++)
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                    }
                }
            }

            p = np; // q.e.d.
            f = nf;
            t = nt;
        }

        if (!taxis && draw_piano)
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

            unsigned F1 = fa.getFrequency( (float)clippedFrustum[i][2] );
            unsigned F2 = fa.getFrequency( (float)clippedFrustum[j][2] );
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

                float u = (ff - clippedFrustum[i][2])/v[2];
                float un = (ff+wN - clippedFrustum[i][2])/v[2];
                float up = (ff-wP - clippedFrustum[i][2])/v[2];
                GLvector pt = clippedFrustum[i]+v*u;
                GLvector pn = clippedFrustum[i]+v*un;
                GLvector pp = clippedFrustum[i]+v*up;

                glPushMatrixContext push_model( GL_MODELVIEW );

                float xscale = 0.016000f;
                float blackw = 0.4f;

                if (sign>0)
                    glTranslatef( xscale*ST, 0.f, 0.f );

                tvector<4,GLfloat> keyColor(0,0,0, 0.7f * blackKey);
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

    glEnable(GL_DEPTH_TEST);
    glDepthMask(true);
}

template<typename T> void glVertex3v( const T* );

template<> void glVertex3v( const GLdouble* t ) {    glVertex3dv(t); }
template<>  void glVertex3v( const GLfloat* t )  {    glVertex3fv(t); }

void Renderer::
        drawFrustum(float alpha)
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
    glBlendFunc(GL_DST_COLOR, GL_SRC_COLOR);
    glColor4f(alpha, alpha, alpha, alpha);
    glBegin( GL_TRIANGLE_FAN );
        for ( std::vector<GLvector>::const_iterator i = clippedFrustum.begin();
                i!=clippedFrustum.end();
                i++)
        {
            //float s = (closest-camera).dot()/(*i-camera).dot();
            //glColor4f(0,0,0,s*.25f);
            glVertex3v( i->v );
        }
    glEnd();

    //glColor4f(0,0,0,.85);
    glBegin( GL_LINE_LOOP );
        for ( std::vector<GLvector>::const_iterator i = clippedFrustum.begin();
                i!=clippedFrustum.end();
                i++)
        {
            glVertex3v( i->v );
        }
    glEnd();
    glDisable(GL_BLEND);
}

} // namespace Heightmap
