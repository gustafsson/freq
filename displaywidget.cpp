#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>

#include <list>
#include <boost/foreach.hpp>

using namespace std;

int DisplayWidget::lastKey = 0;

DisplayWidget::DisplayWidget( boost::shared_ptr<Spectrogram> spectrogram, int timerInterval ) : QGLWidget( ),
  _spectrogram( spectrogram ),
  _px(0), _py(0), _pz(0),
  _rx(0), _ry(0), _rz(0),
  _qx(0), _qy(0), _qz(0),
  _prevX(0), _prevY(0),
  _enqueueGcDisplayList( false )
{
    timeOut();

    if( timerInterval == 0 )
        _timer = 0;
    else
    {
        _timer = new QTimer( this );
        connect( _timer, SIGNAL(timeout()), this, SLOT(timeOutSlot()) );
        _timer->start( timerInterval );
    }
}

DisplayWidget::~DisplayWidget()
{}


void DisplayWidget::mousePressEvent ( QMouseEvent * e )
{
    _prevX = e->x(),
    _prevY = e->y();
}


void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
    float rs = 0.1,
          ps = 0.002;

    int dx = e->x() - _prevX,
        dy = e->y() - _prevY,
        d = dx-dy;
    _prevX = e->x(),
    _prevY = e->y();

    switch( lastKey ) {
        case 'A': _px += d*ps; break;
        case 'S': _py += d*ps; break;
        case 'D': _pz += d*ps; break;
        case 'Q': _rx += d*rs; break;
        case 'W': _ry += d*rs; break;
        case 'E': _rz += d*rs; break;
        case 'Z': _qx += d*ps; break;
        case 'X': _qy += d*ps; break;
        case 'C': _qz += d*ps; break;
        default:
            _ry += dx*rs;
            _rx += dy*rs;
            break;
    }

    glDraw();
}


void DisplayWidget::timeOut()
{
    try{
    } catch (...) {
        string x32= "blaj";
    }
}


void DisplayWidget::timeOutSlot()
{
        timeOut();
}


void DisplayWidget::initializeGL()
{
    glShadeModel(GL_SMOOTH);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
//    glDepthFunc(GL_NEVER);

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    GLfloat LightAmbient[]= { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };
    glShadeModel(GL_SMOOTH);
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
    glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
}


void DisplayWidget::resizeGL( int width, int height ) {
    height = height?height:1;

    glViewport( 0, 0, (GLint)width, (GLint)height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void DisplayWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef( 0, 0, -3 );

    glBegin(GL_LINE_STRIP);
            glColor3f(0,0,0);         glVertex3f( 0, 0, 0 );
            glColor3f(1,0,0);         glVertex3f( _px, _py, _pz );
    glEnd();

    glTranslatef( _px, _py, _pz );

    glRotatef( _rx, 1, 0, 0 );
    glRotatef( _ry, 0, 1, 0 );
    glRotatef( _rz, 0, 0, 1 );

    drawArrows();

    //glTranslatef(-1.5f,0.0f,-6.0f);
    glTranslatef( _qx, _qy, _qz );

    //drawColorFace();

    glPushMatrix();
        glTranslatef( 0, 0, 6 );
        drawWaveform(_spectrogram->transform()->original_waveform());
    glPopMatrix();

    drawSpectrogram();

    //drawWaveform(_spectrogram->getInverseWaveform());
}


void DisplayWidget::drawArrows()
{
    glBegin(GL_LINE_STRIP);
            glColor3f(1,0,0);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.9f, 0.1f, 0.0f);
            glVertex3f( 0.9f, -0.1f, 0.0f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.9f, 0.0f, 0.1f);
            glVertex3f( 0.9f, 0.0f, -0.1f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(0,1,0);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.0f, 0.9f, 0.1f);
            glVertex3f( 0.0f, 0.9f, -0.1f);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.1f, 0.9f, 0.0f);
            glVertex3f( -0.1f, 0.9f, 0.0f);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(0,0,1);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.0f, 0.1f, 0.9f);
            glVertex3f( 0.0f, -0.1f, 0.9f);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.1f, 0.0f, 0.9f);
            glVertex3f( -0.1f, 0.0f, 0.9f);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(1,1,1);
            glVertex3f( _qx, _qy, _qz );
    glEnd();
}


void DisplayWidget::drawColorFace()
{
    glBegin(GL_TRIANGLE_FAN);
            glColor3f( 0, 0, 0);    glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
            glColor3f( 0, 1, 0);    glVertex3f(1.0f,-1.0f, 0.0f);
            glColor3f( 0, 0, 1);    glVertex3f( 1.0f,1.0f, 0.0f);
            glColor3f( .7, .7, 0);    glVertex3f( -1.0f,1.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
    glEnd();
}


int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}


void setWavelengthColor( float wavelengthScalar ) {
    const float spectrum[][3] = {
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }};

    unsigned count = sizeof(spectrum)/sizeof(spectrum[0]);
    float f = count*wavelengthScalar;
    unsigned i = clamp(f, count-1);
    unsigned j = clamp(f+1, count-1);
    float t = f-i;

    GLfloat rgb[] = {  spectrum[i][0]*(1-t) + spectrum[j][0]*t,
                       spectrum[i][1]*(1-t) + spectrum[j][1]*t,
                       spectrum[i][2]*(1-t) + spectrum[j][2]*t
                   };
    glColor3fv( rgb );
}


void DisplayWidget::drawWaveform(pWaveform waveform)
{
    pWaveform_chunk chunk = waveform->getChunk( 0, waveform->number_of_samples(), 0, Waveform_chunk::Only_Real );

    draw_glList<Waveform_chunk>( chunk, DisplayWidget::drawWaveform_chunk_directMode );
}


void DisplayWidget::drawWaveform_chunk_directMode( pWaveform_chunk chunk)
{
    cudaExtent n = chunk->waveform_data->getNumberOfElements();
    const float* data = chunk->waveform_data->getCpuMemory();

    n.height = 1;
    float ifs = 10./chunk->sample_rate; // step per sample
    float max = 1e-6;
    for (unsigned c=0; c<n.height; c++)
    {
        for (unsigned t=0; t<n.width; t++)
            if (fabsf(data[t + c*n.width])>max)
                max = fabsf(data[t + c*n.width]);
    }
    float s = 1/max;

    for (unsigned c=0; c<n.height; c++)
    {
        glTranslatef(0, 0, -.5); // different channels along y
        glBegin(GL_LINE_STRIP);
            glColor3f(1-c,c,0);
            for (unsigned t=0; t<n.width; t++) {
                glVertex3f( -ifs*n.width/2 + ifs*t, s*data[t + c*n.width], 0);

                if (fabsf(data[t + c*n.width])>max)
                    max = fabsf(data[t + c*n.width]);
            }
        glEnd();
    }
}

typedef tvector<3,GLdouble> GLvector;

template<typename f>
GLvector gluProject(tvector<3,f> obj, const GLdouble* model, const GLdouble* proj, const GLint *view) {
    GLvector win;
    gluUnProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]);
    return win;
}

template<typename f>
GLvector gluUnProject(tvector<3,f> win, const GLdouble* model, const GLdouble* proj, const GLint *view) {
    GLvector obj;
    gluUnProject(win[0], win[1], win[2], model, proj, view, &obj[0], &obj[1], &obj[2]);
    return obj;
}

template<typename f>
GLvector gluProject(tvector<3,f> obj) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    return gluProject(obj, model, proj, view);
}

template<typename f>
GLvector gluUnProject(tvector<3,f> win) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    return gluUnProject(win, model, proj, view);
}


void DisplayWidget::drawSpectrogram()
{
//    boost::shared_ptr<Spectrogram_chunk> transform = _spectrogram->getWavelettTransform();

    // find camera position
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    GLvector cam = gluUnProject(GLvector(view[2]/2, view[3]/2, 0), model, proj, view);


    // find units per pixel right beneath camera
    GLvector camToPlane(cam[0],0,cam[2]);
    GLvector sC      = gluProject(camToPlane                  , model, proj, view);
    GLvector sTime = gluProject(camToPlane + GLvector(1,0,0), model, proj, view);
    GLvector sFreq   = gluProject(camToPlane + GLvector(0,0,1), model, proj, view);

    GLdouble timePerPixel = (sTime-sC).rlength(); // == 1/pixelsPerSecond, pixelsPerSecond = (sSample-sC).length()
    GLdouble freqPerPixel = (sFreq-sC).rlength();

    Spectrogram::Reference ref = _spectrogram->findReference(Spectrogram::Position(cam[0], cam[2]), Spectrogram::Position(timePerPixel, freqPerPixel));

    // This is the highest resolution that will be rendered of the dataset
    renderSpectrogramRef(ref);

    // Render its parent and parent and parent until a the largest section is found or a section is found that covers the entire viewed area
    renderParentSpectrogramRef( ref );


//    TODO: use display lists
//    if (_enqueueGcDisplayList)
//        gcDisplayList();
}

void DisplayWidget::renderSpectrogramRef( Spectrogram::Reference ref )
{
    Spectrogram::Position a, b;
    ref.getArea( a, b );
    glBegin(GL_LINE_LOOP );
        glVertex3f( a.time, 0, a.scale );
        glVertex3f( a.time, 0, b.scale );
        glVertex3f( b.time, 0, a.scale );
        glVertex3f( b.time, 0, b.scale );
    glEnd();
    // TODO set the modelview matrix with translate and scale and render the chunk in a unit box
//    Spectrum::Chunk chunk = _spectrogram->getChunk( ref );
//    draw_glList<Spectrogram::Chunk>( chunk, DisplayWidget::drawSpectrogram_chunk_directMode );
}

bool DisplayWidget::renderChildrenSpectrogramRef( Spectrogram::Reference ref )
{
    if (!ref.valid())
        return false;

    Spectrogram::Position a, b;
    ref.getArea( a, b );

    GLvector s00 = gluProject( GLvector( a.time, 0, a.scale) );
    GLvector s01 = gluProject( GLvector( a.time, 0, b.scale) );
    GLvector s10 = gluProject( GLvector( b.time, 0, a.scale) );
    GLvector s11 = gluProject( GLvector( b.time, 0, b.scale) );

    if (0/* does the quadrilateral (s00, s01, s10, s11) not intersect with the screen?*/)
        return false;


    GLdouble t1 = (s00-s10).length();
    GLdouble t2 = (s01-s11).length();
    GLdouble f1 = (s00-s01).length();
    GLdouble f2 = (s10-s11).length();

    bool needBetterF = max(f1,f2) > _spectrogram->scales_per_block(),
        needBetterT = max(t1,t2) > _spectrogram->samples_per_block();
    if ( needBetterF && needBetterT) {
        renderChildrenSpectrogramRef( ref.top().left() );
        renderChildrenSpectrogramRef( ref.top().right() );
        renderChildrenSpectrogramRef( ref.bottom().left() );
        renderChildrenSpectrogramRef( ref.bottom().right() );
    }
    else if ( needBetterF ) {
        renderChildrenSpectrogramRef( ref.top() );
        renderChildrenSpectrogramRef( ref.bottom() );
    }
    else if ( needBetterT ) {
        renderChildrenSpectrogramRef( ref.left() );
        renderChildrenSpectrogramRef( ref.right() );
    }
    else {
        renderSpectrogramRef( ref );
    }

    return true;
}

void DisplayWidget::renderParentSpectrogramRef( Spectrogram::Reference ref )
{
    // Assume that ref has already been drawn, draw sibblings, and call renderParent again
    if (renderChildrenSpectrogramRef( ref.sibbling1() )
        || renderChildrenSpectrogramRef( ref.sibbling2() )
        || renderChildrenSpectrogramRef( ref.sibbling3() ))
    {
        renderParentSpectrogramRef( ref.parent() );
    }
}

void DisplayWidget::drawSpectrogram_block_directMode( Spectrogram::pBlock block )
{
    // TODO: draw within an unit cube
    cudaExtent n = block->transform_data->getNumberOfElements();
    const float* data = block->transform_data->getCpuMemory();

    float ifs = 10./block->sample_rate; // step per sample

    glTranslatef(0, 0, -(2-1)*0.5); // different channels along y

    /* static */ float max = 1;
    float s = 1/max;
    max = 0;
    int fstep = 1;
    int tstep = 10;
    float depthScale = 5.f/n.height;

    glEnable(GL_NORMALIZE);
    for (unsigned fi=0; fi+fstep<n.height-100; fi+=fstep)
    {
        glBegin(GL_TRIANGLE_STRIP);
            float v[3][4] = {{0}};

            int tmax = n.width>>1;
            for (int t=-1; t<=tmax; t+=tstep)
            {
                for (unsigned dt=0; dt<2; dt++)
                    for (unsigned df=0; df<4; df++)
                        v[dt][df] = v[dt+1][df];
                for (unsigned df=0; df<4; df++) {
                    float real = data[clamp(t, tmax-1)*2  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];
                    float complex = data[clamp(t, tmax-1)*2+1  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];

                    //float phase = atan2(complex, real);
                    float amplitude = sqrtf(real*real+complex*complex);
                    v[2][df] = amplitude;
                    v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);

                    //v[2][df] = real;
                }

                if (0>t)
                    continue;

                setWavelengthColor( s*v[1][1] );
                float dt=(v[2][1]-v[0][1]);
                float df=(v[1][2]-v[1][0]);
                glNormal3f( -dt, 2, -df );
                glVertex3f( ifs*t - ifs*tmax/2, s*v[1][1], fi*depthScale);

                setWavelengthColor( s*v[1][2] );
                dt=(v[2][2]-v[0][2]);
                df=(v[1][3]-v[1][1]);
                glNormal3f( -dt, 2, -df );
                glVertex3f( ifs*t - ifs*tmax/2, s*v[1][2], (fi+fstep)*depthScale);

                if (fabsf(v[1][1])>max)
                    max = fabsf(v[1][1]);
            }
        glEnd();
    }
    glDisable(GL_NORMALIZE);
}


template<typename RenderData>
void DisplayWidget::draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ) )
{
    std::map<void*, ListCounter>::iterator itr = _chunkGlList.find(chunk.get());
    if (_chunkGlList.end() == itr) {
        ListCounter cnt;
        cnt.age = ListCounter::Age_JustCreated;
        cnt.displayList = glGenLists(1);

        if (0 != cnt.displayList) {
            glNewList(cnt.displayList, GL_COMPILE_AND_EXECUTE );
            renderFunction( chunk );
            glEndList();
            _chunkGlList[chunk.get()] = cnt;

        } else {
            // render anyway, but not into display list and enqueue gc
            _enqueueGcDisplayList = true;
            renderFunction( chunk );
        }

    } else {
        itr->second.age = ListCounter::Age_InUse; // don't remove

        glCallList( itr->second.displayList );
    }
}

void DisplayWidget::gcDisplayList()
{
    /* remove those display lists that haven't been used since last gc
       (used by draw_glList) */
    for (std::map<void*, ListCounter>::iterator itr = _chunkGlList.begin();
         _chunkGlList.end() != itr;
         ++itr)
    {
        if (ListCounter::Age_ProposedForRemoval == itr->second.age) {
            glDeleteLists( itr->second.displayList, 1 );
            _chunkGlList.erase(itr);
            /* restart for-loop as iterators are invalidated by 'erase' */
            itr = _chunkGlList.begin();
        }
    }

    /* at next gc, remove those that haven't been used since this gc */
    typedef pair<void* const,ListCounter> lcp;
    BOOST_FOREACH( lcp& cnt, _chunkGlList)
    {
/*    for (std::map<Spectrogram_chunk*, ListCounter>::iterator itr = _chunkGlList.begin();
         _chunkGlList.end() != itr;
         ++itr)
    {*/
        cnt.second.age = ListCounter::Age_ProposedForRemoval;
    }

    _enqueueGcDisplayList = false;
}
