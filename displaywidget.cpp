#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>

#include <algorithm>
#include <boost/foreach.hpp>

using namespace std;

int DisplayWidget::lastKey = 0;

DisplayWidget::DisplayWidget( boost::shared_ptr<Spectrogram> spectrogram, int timerInterval )
: QGLWidget( ),
  _renderer( new SpectrogramRenderer( spectrogram )),
  _px(0), _py(0), _pz(-3),
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
        drawWaveform(_renderer->spectrogram()->transform()->original_waveform());
    glPopMatrix();

    _renderer->draw();

    if (_enqueueGcDisplayList)
//        gcDisplayList();
    { ; }

    if (0 < this->_renderer->spectrogram()->read_unfinished_count())
        update();
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



void DisplayWidget::drawWaveform(pWaveform waveform)
{
    static pWaveform_chunk chunk = waveform->getChunk( 0, waveform->number_of_samples(), 0, Waveform_chunk::Only_Real );

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
