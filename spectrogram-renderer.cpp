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
    _initialized(false)
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
            *pos++ = u*2.0f-1.0f;
            *pos++ = 0.0f;
            *pos++ = v*2.0f-1.0f;
            *pos++ = 1.0f;
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


typedef tvector<3,GLdouble> GLvector;

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

bool inFrontOfCamera( GLvector obj ) {
    GLdouble m[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    GLvector v;
    for (int i=0; i<3; i++) {
        GLdouble a = m[4*3+i];
        for (int j=0; j<3; j++)
            a+= obj[j]*m[i + 4*j];
        v[i] = a;
    }
    return v[2] < 0;
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
    if (wStart[0]<0) wStart[0]=0;
    if (wStart[2]<0) wStart[2]=0;
    if (wStart[0]>_spectrogram->transform()->original_waveform()->length()) wStart[0]=_spectrogram->transform()->original_waveform()->length();
    if (wStart[2]>1) wStart[2]=1;

    GLvector sBase      = gluProject(wStart);
    sBase[2] = 0.1;

    if (!validWindowPos(sBase)) {
        // point not visible, take bottom of screen instead
        if (wCam[1] > 0)
            sBase = GLvector(view[0]+view[2]/2, view[1], 0.1);
        else // or top of screen if beneath
            sBase = GLvector(view[0]+view[2]/2, view[1]+view[3], 0.1);
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

    Spectrogram::Reference ref = _spectrogram->findReference(Spectrogram::Position(wBase[0], wBase[2]), Spectrogram::Position(timePerPixel, freqPerPixel));

    // This is the starting point for rendering the dataset
    renderChildrenSpectrogramRef(ref);

    // Render its parent and parent and parent until a the largest section is found or a section is found that covers the entire viewed area
    renderParentSpectrogramRef( ref );

    GlException_CHECK_ERROR();
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

    Spectrogram::Position p[2];
    ref.getArea( p[0], p[1] );

    GLvector corner[4];
    for (int i=0; i<4; i++)
        corner[i] = GLvector( p[i/2].time, 0, p[i%2].scale);

    bool valid[4], visible = false;
    for (int i=0; i<4; i++)
        visible |= (valid[i] = inFrontOfCamera( corner[i] ));

    if (!visible)
        return false;

    GLvector screen[4];
    for (int i=0; i<4; i++) {
        if (!valid[i]) continue;
        screen[i] = gluProject( corner[i] );
    }

    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);
    bool above=true, under=true, right=true, left=true;
    for (int i=0; i<4; i++) {
        if (!valid[i]) continue;
        if (screen[i][0] > view[0]) left=false;
        if (screen[i][0] < view[0]+view[2]) right=false;
        if (screen[i][1] > view[1]) under=false;
        if (screen[i][1] < view[1]+view[3]) above=false;
    }

    if (above || under || right || left)
        return false;

    GLdouble t1 = valid[0] && valid[2] ? (screen[0]-screen[2]).length() : 0;
    GLdouble t2 = valid[1] && valid[3] ? (screen[1]-screen[3]).length() : 0;
    GLdouble f1 = valid[0] && valid[1] ? (screen[0]-screen[1]).length() : 0;
    GLdouble f2 = valid[2] && valid[3] ? (screen[2]-screen[3]).length() : 0;

    GLdouble needBetterF, needBetterT;
    if (0==f1 && 0==f2) needBetterF = 1.1;
    else needBetterF = max(f1,f2) / _spectrogram->scales_per_block();
    if (0==t1 && 0==t2) needBetterT = 1.1;
    else needBetterT = max(t1,t2) / _spectrogram->samples_per_block();

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

