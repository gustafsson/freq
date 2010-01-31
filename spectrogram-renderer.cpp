#include "GL/glew.h"
#include <stdio.h>
#include "spectrogram-renderer.h"
#include "spectrogram-vbo.h"
#include <list>
#include <GlException.h>
using namespace std;

SpectrogramRenderer::SpectrogramRenderer( pSpectrogram spectrogram )
:   _spectrogram(spectrogram),
    _mesh_index_buffer(0),
    _mesh_width(0),
    _mesh_height(0),
    _initialized(false),
    _redundancy(2) // 1 means every pixel gets its own vertex, 10 means every 10th pixel gets its own vertex
{
}

void SpectrogramRenderer::setSize( unsigned w, unsigned h)
{
    if (w == _mesh_width && h ==_mesh_height)
        return;

    createMeshIndexBuffer(w, h);
    createMeshPositionVBO(w, h);
}

// create index buffer for rendering quad mesh
void SpectrogramRenderer::createMeshIndexBuffer(unsigned w, unsigned h)
{
    // create index buffer
    if (_mesh_index_buffer)
        glDeleteBuffersARB(1, &_mesh_index_buffer);

    _mesh_width = w;
    _mesh_height = h;

    int size = ((w*2)+2)*(h-1)*sizeof(GLuint);
    glGenBuffersARB(1, &_mesh_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!indices) {
        return;
    }

    for(unsigned y=0; y<h-1; y++) {
        for(unsigned x=0; x<w; x++) {
            *indices++ = y*w+x;
            *indices++ = (y+1)*w+x;
        }
        // start new strip with degenerate triangle
        *indices++ = (y+1)*w+(w-1);
        *indices++ = (y+1)*w;
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void SpectrogramRenderer::createMeshPositionVBO(unsigned w, unsigned h)
{
    _mesh_position.reset( new Vbo( w*h*4*sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
    float *pos = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!pos) {
        return;
    }

    for(unsigned y=0; y<h; y++) {
        for(unsigned x=0; x<w; x++) {
            float u = x / (float) (w-1);
            float v = y / (float) (h-1);
            *pos++ = u;
            *pos++ = 0.0f;
            *pos++ = v;
            *pos++ = 1.0f;
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


typedef tvector<3,GLdouble> GLvector;
typedef tvector<4,GLdouble> GLvector4;

template<typename f>
GLvector gluProject(tvector<3,f> obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
    GLvector win;
    bool s = (GLU_TRUE == gluProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]));
    if(r) *r=s;
    return win;
}

template<typename f>
GLvector gluUnProject(tvector<3,f> win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
    GLvector obj;
    bool s = (GLU_TRUE == gluUnProject(win[0], win[1], win[2], model, proj, view, &obj[0], &obj[1], &obj[2]));
    if(r) *r=s;
    return obj;
}

template<typename f>
GLvector gluProject(tvector<3,f> obj, bool *r=0) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    return gluProject(obj, model, proj, view, r);
}

template<typename f>
GLvector gluUnProject(tvector<3,f> win, bool *r=0) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    return gluUnProject(win, model, proj, view, r);
}

bool validWindowPos(GLvector win) {
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);

    return win[0]>view[0] && win[1]>view[1]
            && win[0]<view[0]+view[2]
            && win[1]<view[1]+view[3]
            && win[2]>=0.1 && win[2]<=100;
}

GLvector applyModelMatrix(GLvector obj) {
    GLdouble m[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    GLvector eye;
    for (int i=0; i<3; i++) {
        GLdouble a = m[4*3+i];
        for (int j=0; j<3; j++)
            a+= obj[j]*m[i + 4*j];
        eye[i] = a;
    }
    return eye;
}

GLvector applyProjectionMatrix(GLvector eye) {
    GLdouble p[16];
    glGetDoublev(GL_PROJECTION_MATRIX, p);
    GLvector clip;
    for (int i=0; i<3; i++) {
        GLdouble a = p[4*3+i];
        for (int j=0; j<3; j++)
            a+= eye[j]*p[i + 4*j];
        clip[i] = a;
    }
    return clip;
}

bool inFrontOfCamera( GLvector obj ) {
    GLvector eye = applyModelMatrix(obj);

    GLvector clip = applyProjectionMatrix(eye);

    return clip[2] > 0.1;
}

class glPushMatrixContext
{
public:
    glPushMatrixContext() { glPushMatrix(); }
    ~glPushMatrixContext() { glPopMatrix(); }
};

void SpectrogramRenderer::init()
{
    // initialize necessary OpenGL extensions
    GlException_CHECK_ERROR_MSG("1");

    if (0 != glewInit() ) {
        fprintf(stderr, "ERROR: Couldn't initialize \"glew\".");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }

    if (! glewIsSupported("GL_VERSION_2_0" )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }

    if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 1.5\n");
            fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
            fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            fflush(stderr);
            exit(-1);
    }

    // load shader
    _shader_prog = loadGLSLProgram("spectrogram.vert", "spectrogram.frag");

    setSize( _spectrogram->samples_per_block(), _spectrogram->scales_per_block() );

    _initialized=true;

    GlException_CHECK_ERROR_MSG("2");
}

void SpectrogramRenderer::draw()
{
    if (!_initialized) init();

    glPushMatrixContext();

    glScalef( 10, 1, 5 );

//    boost::shared_ptr<Spectrogram_chunk> transform = _spectrogram->getWavelettTransform();

    // find camera position
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);

    GLvector wCam = gluUnProject(GLvector(view[2]/2, view[3]/2, -1e100) ); // approx==camera position

    // as we
    // 1) never tilt the camera around the z-axes
    // 2) x-rotation is performed before y-rotation
    // 3) never rotate more than 90 degrees around x
    // => the closest point in the plane from the camera is on the line: (view[2]/2, view[3])-(view[2]/2, view[3]/2)
    // Check if the point right beneath the camera is visible

    // find units per pixel right beneath camera
    // wC is the point on screen that renders the closest pixel of the plane

    GLvector wStart = GLvector(wCam[0],0,wCam[2]);
    float length = _spectrogram->transform()->original_waveform()->length();
    if (wStart[0]<0) wStart[0]=0;
    if (wStart[2]<0) wStart[2]=0;
    if (wStart[0]>length) wStart[0]=length;
    if (wStart[2]>1) wStart[2]=1;

    GLvector sBase      = gluProject(wStart);
    sBase[2] = 0.1;

    if (!validWindowPos(sBase)) {
        // point not visible => the pixel closest to the plane is somewhere along the border of the screen
        // take bottom of screen instead
        if (wCam[1] > 0)
            sBase = GLvector(view[0]+view[2]/2, view[1], 0.1);
        else // or top of screen if beneath
            sBase = GLvector(view[0]+view[2]/2, view[1]+view[3], 0.1);

/*        GLvector wBase = gluUnProject( sBase ),
        if (wBase[0]<0 || )
                        sBase = GLvector(view[0]+view[2]/2, view[1]+view[3], 0.1);
wStart[0]=0;
        if (wStart[2]<0) wStart[2]=0;
        if (wStart[0]>length) wStart[0]=length;
        if (wStart[2]>1) wStart[2]=1;*/
    }

    // find world coordinates of projection surface
    GLvector
            wBase = gluUnProject( sBase ),
            w1 = gluUnProject(sBase + GLvector(1,0,0) ),
            w2 = gluUnProject(sBase + GLvector(0,1,0) );

    // directions
    GLvector
            dirBase = wBase-wCam,
            dir1 = w1-wCam,
            dir2 = w2-wCam;

    // valid projection on xz-plane exists if dir?[1]<0 wBase[1]<0
    GLvector
            xzBase = wCam - dirBase*(wCam[1]/dirBase[1]),
            xz1 = wCam - dir1*(wCam[1]/dir1[1]),
            xz2 = wCam - dir2*(wCam[1]/dir2[1]);

    // compute {units in xz-plane} per {screen pixel}, that determines the required resolution
    GLdouble
            timePerPixel = 0,
            freqPerPixel = 0;

    if (dir1[1] != 0 && dir2[1] != 0) {
        timePerPixel = max(timePerPixel, fabs(xz1[0]-xz2[0]));
        freqPerPixel = max(freqPerPixel, fabs(xz1[2]-xz2[2]));
    }
    if (dir1[1] != 0 && dirBase[1] != 0) {
        timePerPixel = max(timePerPixel, fabs(xz1[0]-xzBase[0]));
        freqPerPixel = max(freqPerPixel, fabs(xz1[2]-xzBase[2]));
    }
    if (dir2[1] != 0 && dirBase[1] != 0) {
        timePerPixel = max(timePerPixel, fabs(xz2[0]-xzBase[0]));
        freqPerPixel = max(freqPerPixel, fabs(xz2[2]-xzBase[2]));
    }

    if (0 == timePerPixel)
        timePerPixel = max(fabs(w1[0]-wBase[0]), fabs(w2[0]-wBase[0]));
    if (0 == freqPerPixel)
        freqPerPixel = max(fabs(w1[2]-wBase[2]), fabs(w2[2]-wBase[2]));

    if (0==freqPerPixel) freqPerPixel=timePerPixel;
    if (0==timePerPixel) timePerPixel=freqPerPixel;
    timePerPixel*=_redundancy;
    freqPerPixel*=_redundancy;

    Spectrogram::Reference ref = _spectrogram->findReference(Spectrogram::Position(wBase[0], wBase[2]), Spectrogram::Position(timePerPixel, freqPerPixel));

    Spectrogram::Position mss = _spectrogram->max_sample_size();
    ref = _spectrogram->findReference(Spectrogram::Position(0,0), mss);

    beginVboRendering();

    renderChildrenSpectrogramRef(ref);

/*    // This is the starting point for rendering the dataset
    while (false==renderChildrenSpectrogramRef(ref) && !ref.parent().toLarge() )
        ref = ref.parent();



    // Render its parent and parent and parent until a the largest section is found or a section is found that covers the entire viewed area
    renderParentSpectrogramRef( ref );
*/
    endVboRendering();

    GlException_CHECK_ERROR();
}

void SpectrogramRenderer::beginVboRendering()
{
    unsigned meshW = _spectrogram->samples_per_block();
    unsigned meshH = _spectrogram->scales_per_block();

    glUseProgram(_shader_prog);

    // Set default uniform variables parameters for the vertex shader
    GLuint uniHeightScale, uniChopiness, uniSize;

    uniHeightScale = glGetUniformLocation(_shader_prog, "heightScale");
    glUniform1f(uniHeightScale, 0.5f);

    uniChopiness   = glGetUniformLocation(_shader_prog, "chopiness");
    glUniform1f(uniChopiness, 1.0f);

    uniSize        = glGetUniformLocation(_shader_prog, "size");
    glUniform2f(uniSize, meshW, meshH);

    // Set default uniform variables parameters for the pixel shader
    GLuint uniDeepColor, uniShallowColor, uniSkyColor, uniLightDir;

    uniDeepColor = glGetUniformLocation(_shader_prog, "deepColor");
    glUniform4f(uniDeepColor, 0.0f, 0.0f, 0.1f, 1.0f);

    uniShallowColor = glGetUniformLocation(_shader_prog, "shallowColor");
    glUniform4f(uniShallowColor, 0.1f, 0.4f, 0.3f, 1.0f);

    uniSkyColor = glGetUniformLocation(_shader_prog, "skyColor");
    glUniform4f(uniSkyColor, 0.5f, 0.5f, 0.5f, 1.0f);

    uniLightDir = glGetUniformLocation(_shader_prog, "lightDir");
    glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);
    // end of uniform settings

    glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);
}

void SpectrogramRenderer::endVboRendering() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}

bool SpectrogramRenderer::renderSpectrogramRef( Spectrogram::Reference ref )
{
    if (!ref.containsSpectrogram())
        return false;

    Spectrogram::Position a, b;
    ref.getArea( a, b );
    glPushMatrix();
    if (1) { // matrix push/pop
        glTranslatef(a.time, 0, a.scale);
        glScalef(b.time-a.time, 1, b.scale-a.scale);

        Spectrogram::pBlock block = _spectrogram->getBlock( ref );
        if (0!=block.get()) {
            if (0 /* direct rendering */ )
                block->vbo->draw_directMode();
            else if (1 /* vbo */ )
                block->vbo->draw( this );

        } else {
            // getBlock would try to find something else if the requested block wasn't readily available.
            // If getBlock fails, we're out of memory. Indicate by not drawing the surface but only a wireframe

            glBegin(GL_LINE_LOOP );
                glVertex3f( 0, 0, 0 );
                glVertex3f( 0, 0, 1 );
                glVertex3f( 1, 0, 0 );
                glVertex3f( 1, 0, 1 );
            glEnd();
        }
    }
    glPopMatrix();

    return true;
}

bool SpectrogramRenderer::renderChildrenSpectrogramRef( Spectrogram::Reference ref )
{
    if (!ref.containsSpectrogram())
        return false;

    float timePixels, scalePixels;
    if (!computePixelsPerUnit( ref, timePixels, scalePixels))
        return false;

    GLdouble needBetterF, needBetterT;

    if (0==scalePixels)
        needBetterF = 1.01;
    else
        needBetterF = scalePixels / (_redundancy*_spectrogram->scales_per_block());
    if (0==timePixels)
        needBetterT = 1.01;
    else
        needBetterT = timePixels / (_redundancy*_spectrogram->samples_per_block());

    if ( needBetterF > needBetterT && needBetterF > 1 && ref.top().containsSpectrogram() ) {
        renderChildrenSpectrogramRef( ref.top() );
        renderChildrenSpectrogramRef( ref.bottom() );
    }
    else if ( needBetterT > 1 && ref.left().containsSpectrogram() ) {
        renderChildrenSpectrogramRef( ref.left() );
        renderChildrenSpectrogramRef( ref.right() );
    }
    else {
        renderSpectrogramRef( ref );
    }

    return true;
}

void SpectrogramRenderer::renderParentSpectrogramRef( Spectrogram::Reference ref )
{
    // Assume that ref has already been drawn, draw sibblings, and call renderParent again
    renderChildrenSpectrogramRef( ref.sibbling1() );
    renderChildrenSpectrogramRef( ref.sibbling2() );
    renderChildrenSpectrogramRef( ref.sibbling3() );

    if (!ref.parent().toLarge() )
        renderParentSpectrogramRef( ref.parent() );
}

// the normal does not need to be normalized, and back/front doesn't matter
static float linePlane( GLvector planeNormal, GLvector pt, GLvector dir ) {
    return -(pt % planeNormal)/ (dir % planeNormal);
}

static GLvector planeIntersection( GLvector planeNormal, GLvector pt, GLvector dir ) {
    return pt + dir*linePlane(planeNormal, pt, dir);
}

GLvector xzIntersection( GLvector pt1, GLvector pt2 ) {
    return planeIntersection( GLvector(0,1,0), pt1, pt2-pt1 );
}

static GLvector cameraIntersection( GLvector pt1, GLvector pt2, float &s ) {
    GLvector s1 = applyProjectionMatrix(applyModelMatrix(pt1));
    GLvector s2 = applyProjectionMatrix(applyModelMatrix(pt2));

    //bs = (.3-s1[2]) / (s2[2]-s1[2]);
    s = (.03-s1[2]) / (s2[2]-s1[2]);
    return pt1 + (pt2-pt1)*s;
}

GLvector xzIntersection2( GLvector pt1, GLvector pt2, float &s ) {
    s = -pt1[1] / (pt2[1]-pt1[1]);
    return pt1 + (pt2-pt1)*s;
}

void updateBounds(float bounds[4], GLvector p) {
    if (bounds[0] > p[0]) bounds[0] = p[0];
    if (bounds[2] < p[0]) bounds[2] = p[0];
    if (bounds[1] > p[1]) bounds[1] = p[1];
    if (bounds[3] < p[1]) bounds[3] = p[1];
}

static void updateBorderLength(double& length, GLvector border, float s) {
    border[2] = 0;
    length = max(length, border.length()/s);
}

bool SpectrogramRenderer::computePixelsPerUnit( Spectrogram::Reference ref, float& timePixels, float& scalePixels )
{
    Spectrogram::Position p[2];
    ref.getArea( p[0], p[1] );

    GLvector corner[4]=
    {
        GLvector( p[0].time, 0, p[0].scale),
        GLvector( p[0].time, 0, p[1].scale),
        GLvector( p[1].time, 0, p[1].scale),
        GLvector( p[1].time, 0, p[0].scale)
    };

    bool valid[4], visible = false;
    for (int i=0; i<4; i++)
        visible |= (valid[i] = inFrontOfCamera( corner[i] ));

    if (!visible)
        return false;

    // Find projection of object coordinates in corner[i] onto window coordinates
    // Compare length in pixels of each side
    // Check if projection is visible
    double b[2]={0};
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);
    bool above=true, under=true, right=true, left=true;
    for (int i=0; i<4; i++)
    {
        int nexti=(i+1)%4;
        GLvector screen[2];
        float s=NAN;

        if (valid[i] && valid[nexti]) {
            screen[0] = gluProject( corner[i] );
            screen[1] = gluProject( corner[nexti] );
            s = 1;

        } else if (valid[i] && !valid[nexti]) {
            GLvector xz = cameraIntersection( corner[i], corner[nexti], s );
            screen[0] = gluProject( corner[i] );
            screen[1] =  gluProject( xz );

        } else if (!valid[i] && valid[nexti]) {
            GLvector xz = cameraIntersection( corner[i], corner[nexti], s );
            screen[0] = gluProject( xz );
            screen[1] = gluProject( corner[nexti] );
        }

        if (isnan(s))
            continue;

        bool labove=true, lunder=true, lright=true, lleft=true;
        for (int j=0; j<2; j++) {
            if (screen[j][0] > view[0]) lleft=false;
            if (screen[j][0] < view[0]+view[2]) lright=false;
            if (screen[j][1] > view[1]) lunder=false;
            if (screen[j][1] < view[1]+view[3]) labove=false;
        }

        above &= labove;
        under &= lunder;
        right &= lright;
        left &= lleft;
        // If everything is
        if (labove || lunder || lright || lleft)
            continue;

        updateBorderLength( b[i%2], screen[0] - screen[1], s );
    }

    // If everything is
    if (above || under || right || left)
        return false;

    scalePixels = b[0];
    timePixels = b[1];

    return true;
}
