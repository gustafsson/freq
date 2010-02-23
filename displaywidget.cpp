#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>

#include <QtGui/QFileDialog>
#include <CudaException.h>

#include <algorithm>
#include <boost/foreach.hpp>

#include <tvector.h>
#include <math.h>
#include <GL/glut.h>
#include <stdio.h>

#ifdef _MSC_VER
#define M_PI 3.1415926535
#endif

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


using namespace std;

float MouseControl::deltaX( float x )
{
  if( down )
    return x - lastx;
  
  return 0;
}
float MouseControl::deltaY( float y )
{
  if( down )
    return y - lasty;
  
  return 0;
}

bool MouseControl::worldPos(GLdouble &ox, GLdouble &oy)
{
  return worldPos(this->lastx, this->lasty, ox, oy);
}
bool MouseControl::worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy)
{
  GLdouble s;
  bool test[2];
  GLvector win_coord, world_coord[2];
  
  win_coord = GLvector(x, y, 0.1);
  
  world_coord[0] = gluUnProject<GLdouble>(win_coord, &test[0]);
  //printf("CamPos1: %f: %f: %f\n", world_coord[0][0], world_coord[0][1], world_coord[0][2]);
  
  win_coord[2] = 0.6;
  world_coord[1] = gluUnProject<GLdouble>(win_coord, &test[1]);
  //printf("CamPos2: %f: %f: %f\n", world_coord[1][0], world_coord[1][1], world_coord[1][2]);
  
  s = (-world_coord[0][1]/(world_coord[1][1]-world_coord[0][1]));
  
  ox = world_coord[0][0] + s * (world_coord[1][0]-world_coord[0][0]);
  oy = world_coord[0][2] + s * (world_coord[1][2]-world_coord[0][2]);
  
  float minAngle = 20;
  if( s < 0 || world_coord[0][1]-world_coord[1][1] < sin(minAngle *(M_PI/180)) * (world_coord[0]-world_coord[1]).length() )
    return false;

  return test[0] && test[1];
}

void MouseControl::press( float x, float y )
{
  touch();
  update( x, y );
  down = true;
}
void MouseControl::update( float x, float y )
{
  touch();
  lastx = x;
  lasty = y;
}
void MouseControl::release()
{
  //touch();
  down = false;
}
bool MouseControl::isTouched()
{
  if(hold == 0)
    return true;
  else
    return false;
}


DisplayWidget* DisplayWidget::gDisplayWidget = 0;

DisplayWidget::DisplayWidget( boost::shared_ptr<Spectrogram> spectrogram, int timerInterval, std::string playback_source_test )
: QGLWidget( ),
  lastKey(0),
  xscale(1),
  _renderer( new SpectrogramRenderer( spectrogram )),
  _px(0), _py(0), _pz(-10),
  _rx(45), _ry(225), _rz(0),
  _qx(0), _qy(0), _qz(3.6f/5),
  _prevX(0), _prevY(0), _targetQ(0),
  _enqueueGcDisplayList( false ),
  selecting(false)
{
#ifdef _WIN32
    int c=1;
    char* dum="dum\0";
    glutInit(&c,&dum);
#else
    int c=0;
    glutInit(&c,0);
#endif
    gDisplayWidget = this;
    float l = _renderer->spectrogram()->transform()->original_waveform()->length();
    _qx = .5 * l;
    selection[0].x = l*.5f;
    selection[0].y = 0;
    selection[0].z = .85f;
    selection[1].x = l*sqrt(2.0f);
    selection[1].y = 0;
    selection[1].z = 2;

    // no selection
    selection[0].x = selection[1].x;
    selection[0].z = selection[1].z;

    _renderer->spectrogram()->transform()->setInverseArea( selection[0].x, selection[0].z, selection[1].x, selection[1].z );

    yscale = Yscale_LogLinear;
    timeOut();

    if ( timerInterval != 0 )
    {
        startTimer(timerInterval);
    }

    if (!playback_source_test.empty())
    {
        open_inverse_test(playback_source_test);
    } else {
        _transform = _renderer->spectrogram()->transform();
    }
}

DisplayWidget::~DisplayWidget()
{}

void DisplayWidget::keyPressEvent( QKeyEvent *e )
{
    lastKey = e->key();
    pTransform t = _renderer->spectrogram()->transform();
    switch (lastKey )
    {
        case ' ':
            _transform->play_inverse();
            break;
        case 'c': case 'C':
        {
            t->recompute_filter(pFilter());
            if( _transform != t )
                _transform->recompute_filter(pFilter());

            BOOST_FOREACH( pFilter f, t->filter_chain )
            {
                float start, end;
                f->range(start,end);
                _renderer->spectrogram()->invalidate_range(start,end);
            }

            t->filter_chain.clear();
            if( _transform != t )
                _transform->filter_chain.clear();

            update();
            break;
        }
        case 'a': case 'A': case '\n': case '\r':
        {
            pFilter f(new EllipsFilter( t->built_in_filter ) );

            t->filter_chain.push_back(f);
            t->recompute_filter(f);
            if( _transform != t ) {
                _transform->filter_chain.push_back(f);
                _transform->recompute_filter(f);
            }

            float start, end;
            f->range(start,end);
            _renderer->spectrogram()->invalidate_range(start,end);
            update();
            break;
        }
        case 'e': case 'E':
        {
            open_inverse_test();
            break;
        }
    }
}

void DisplayWidget::keyReleaseEvent ( QKeyEvent *  )
{
    lastKey = 0;
}

void DisplayWidget::mousePressEvent ( QMouseEvent * e )
{
  switch ( e->button() )
  {
    case Qt::LeftButton:
      if(' '==lastKey)
      	selectionButton.press( e->x(), this->height() - e->y() );
      else
      	leftButton.press( e->x(), this->height() - e->y() );
      //printf("LeftButton: Press\n");
      break;
      
    case Qt::MidButton:
      middleButton.press( e->x(), this->height() - e->y() );
      //printf("MidButton: Press\n");
      break;
      
    case Qt::RightButton:
    {
      rightButton.press( e->x(), this->height() - e->y() );
      //printf("RightButton: Press\n");
    }
      break;
      
    default:
      break;
  }
  
  if(leftButton.isDown() && rightButton.isDown())
	selectionButton.press( e->x(), this->height() - e->y() );
  
  glDraw();
    _prevX = e->x(),
    _prevY = e->y();
}

void DisplayWidget::mouseReleaseEvent ( QMouseEvent * e )
{
  switch ( e->button() )
  {
    case Qt::LeftButton:
      leftButton.release();
      selectionButton.release();
      //printf("LeftButton: Release\n");
      selecting = false;
      break;
      
    case Qt::MidButton:
      middleButton.release();
      //printf("MidButton: Release\n");
      break;
      
    case Qt::RightButton:
      rightButton.release();
      selectionButton.release();
      //printf("RightButton: Release\n");
      break;
      
    default:
      break;
  }
  glDraw();
}

void DisplayWidget::wheelEvent ( QWheelEvent *e )
{
  float ps = 0.0005;
  float rs = 0.08;
  if( e->orientation() == Qt::Horizontal )
  {
    _ry -= rs * e->delta();
  }
  else
  {
    if(e->modifiers().testFlag(Qt::ControlModifier))
        xscale *= (1-ps * e->delta());
    else
        _pz *= (1+ps * e->delta());
    //_pz -= ps * e->delta();

    //_rx -= ps * e->delta();
  }
  
  glDraw();
}

void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
  float rs = 0.2;
  
  int x = e->x(), y = this->height() - e->y();
  
  if (selectionButton.isDown())
  {
    GLdouble p[2];
    if (selectionButton.worldPos(x, y, p[0], p[1]))
    {
      if (!selecting) {
        selection[0].x = selection[1].x = p[0];
        selection[0].y = selection[1].y = 0;
        selection[0].z = selection[1].z = p[1];
        selecting = true;
      } else {
        selection[1].x = p[0];
        selection[1].y = 0;
        selection[1].z = p[1];
        _renderer->spectrogram()->transform()->setInverseArea( selection[0].x, selection[0].z, selection[1].x, selection[1].z );
        if (_transform != _renderer->spectrogram()->transform())
            _transform->setInverseArea( selection[0].x, selection[0].z, selection[1].x, selection[1].z );
      }
    }
  } else {
      //Controlling the rotation with the left button.
      _ry += (1-orthoview)*rs * leftButton.deltaX( x );
      _rx -= rs * leftButton.deltaY( y );
      if (_rx<0) _rx=0;
      if (_rx>90) { _rx=90; orthoview=1; }
      if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }

      //Controlling the the position with the right button.

      if( rightButton.isDown() )
      {
        GLvector last, current;
        if( rightButton.worldPos(last[0], last[1]) &&
            rightButton.worldPos(x, y, current[0], current[1]) )
        {
          float l = _renderer->spectrogram()->transform()->original_waveform()->length();

          _qx -= current[0] - last[0];
          _qz -= current[1] - last[1];

          if (_qx<0) _qx=0;
          if (_qz<0) _qz=0;
          if (_qz>8.f/5) _qz=8.f/5;
          if (_qx>l) _qx=l;
        }
      }
  }
  
  
  //Updating the buttons
  leftButton.update( x, y );
  rightButton.update( x, y );
  middleButton.update( x, y );
  selectionButton.update( x, y );
  
  glDraw();
}


void DisplayWidget::timeOut()
{
  leftButton.untouch();
  middleButton.untouch();
  rightButton.untouch();
  selectionButton.untouch();

  if(selectionButton.isDown() && selectionButton.getHold() == 5)
  {
    _transform->play_inverse();
  }
}

void DisplayWidget::timerEvent(QTimerEvent *)
{
    timeOut();
}


void DisplayWidget::timeOutSlot()
{
    timeOut();
}

void DisplayWidget::open_inverse_test(std::string soundfile)
{
    if (0 == soundfile.length() || !QFile::exists(soundfile.c_str())) {
        QString fileName = QFileDialog::getOpenFileName(0, "Open sound file");
        if (0 == fileName.length()) {
            _transform = _renderer->spectrogram()->transform();
            return;
        }
        soundfile = fileName.toStdString();
    }
    boost::shared_ptr<Waveform> wf( new Waveform( soundfile.c_str() ) );
    boost::shared_ptr<Transform> t = _renderer->spectrogram()->transform();

    _transform.reset();
    _transform.reset( new Transform(wf, 0, t->samples_per_chunk(), t->scales_per_octave(), t->wavelet_std_t() ) );

    try
    {
        _transform->get_inverse_waveform();
    } catch (const CudaException& x) {
        _renderer->spectrogram()->gc();
        _transform->get_inverse_waveform();
    }

    _transform->setInverseArea( selection[0].x, selection[0].z, selection[1].x, selection[1].z );
}

void DisplayWidget::initializeGL()
{
    glShadeModel(GL_SMOOTH);

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_LINE_SMOOTH);
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
    TaskTimer tt(__FUNCTION__);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

/*    glBegin(GL_LINE_STRIP);
            glColor3f(0,0,0);         glVertex3f( v1.x, v1.y, v1.z );
            glColor3f(1,0,0);         glVertex3f( _px, _py, _pz );
    glEnd();
*/
    glTranslatef( _px, _py, _pz );

    glRotatef( _rx, 1, 0, 0 );
    glRotatef( fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( _rz, 0, 0, 1 );

    glScalef(-xscale, 1-.99*orthoview, 5);

    glTranslatef( -_qx, -_qy, -_qz );

    orthoview.TimeStep(.08);

    pWaveform inv, inv2;
    glPushMatrix();
        glTranslatef( 0, 0, 1.25f );
        glScalef(1, 1, .15);
        glColor4f(0,1,0,1);
        {
            TaskTimer tt("drawWaveform( inverse )");
            drawWaveform(inv = _renderer->spectrogram()->transform()->get_inverse_waveform());
            // drawWaveform(inv = _transform->get_inverse_waveform());
            if( _transform != _renderer->spectrogram()->transform() )
            {
                inv2 = _transform->get_inverse_waveform();
                pWaveform_chunk chunk = inv2->getChunkBehind();
                if (chunk->modified) {
                    update();
                    chunk->modified=false;
                }
            }

        }
        glTranslatef( 0, 0, 2.f );
        glColor4f(0,0,0,1);
        {
            TaskTimer tt("drawWaveform( original )");
            drawWaveform(_renderer->spectrogram()->transform()->original_waveform());
        }
    glPopMatrix();

    _renderer->draw();

    static float prev_xscale = xscale;
    draw_glList<SpectrogramRenderer>( _renderer, DisplayWidget::drawSpectrogram_borders_directMode, xscale!=prev_xscale );

    if (_enqueueGcDisplayList)
//        gcDisplayList();
    { ; }

    if (0 < this->_renderer->spectrogram()->read_unfinished_count()) {
        if (inv->getChunkBehind()->play_when_done || inv->getChunkBehind()->modified)
            _renderer->spectrogram()->dont_compute_until_next_read_unfinished_count();
        if (inv2) if (inv2->getChunkBehind()->play_when_done || inv2->getChunkBehind()->modified)
            _renderer->spectrogram()->dont_compute_until_next_read_unfinished_count();
        update();
    }


    drawSelection();
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
    //static pWaveform_chunk chunk = waveform->getChunk( 0, waveform->number_of_samples(), 0, Waveform_chunk::Only_Real );
    pWaveform_chunk chunk = waveform->getChunkBehind();
    if (chunk->modified) {
        chunk->was_modified = true;
        drawWaveform_chunk_directMode( chunk );
        update();
        chunk->modified=false;
    } else if(chunk->was_modified) {
        draw_glList<Waveform_chunk>( chunk, DisplayWidget::drawWaveform_chunk_directMode, true );
        chunk->was_modified = false;
    } else {
        draw_glList<Waveform_chunk>( chunk, DisplayWidget::drawWaveform_chunk_directMode, false );
    }
}

void DisplayWidget::drawWaveform_chunk_directMode( pWaveform_chunk chunk)
{
    TaskTimer tt(__FUNCTION__);
    cudaExtent n = chunk->waveform_data->getNumberOfElements();
    const float* data = chunk->waveform_data->getCpuMemory();

    n.height = 1;
    float ifs = 1./chunk->sample_rate; // step per sample
    float max = 1e-6;
    //for (unsigned c=0; c<n.height; c++)
    {
        unsigned c=0;
        for (unsigned t=0; t<n.width; t++)
            if (fabsf(data[t + c*n.width])>max)
                max = fabsf(data[t + c*n.width]);
    }
    float s = 1/max;

    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    unsigned c=0;
//    for (unsigned c=0; c<n.height; c++)
    {
        glBegin(GL_TRIANGLE_STRIP);
        //glBegin(GL_POINTS);
            for (unsigned t=0; t<n.width; t+=std::max((size_t)1,n.width/2000)) {
                /*float lmin,lmax = (lmin = data[t + c*n.width]);
                for (unsigned j=0; j<std::max((size_t)2, (n.width/1000)) && t<n.width;j++, t++) {
                    const float &a = data[t + c*n.width];
                    if (a<lmin) lmin=a;
                    if (a>lmax) lmax=a;
                }
                glVertex3f( ifs*t, 0, s*lmax);
                glVertex3f( ifs*t, 0, s*lmin);*/
                glVertex3f( ifs*t, 0, s*data[t + c*n.width]);
                float pt = t;
                t+=std::max((size_t)1,n.width/2000);
                if (t<n.width)
                    glVertex3f( ifs*pt, 0, s*data[t + c*n.width]);
            }
        glEnd();
//        glTranslatef(0, 0, -.5); // different channels along y
    }

    glDepthMask(true);
    glDisable(GL_BLEND);
}


template<typename RenderData>
void DisplayWidget::draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw )
{
    std::map<void*, ListCounter>::iterator itr = _chunkGlList.find(chunk.get());
    if (_chunkGlList.end() != itr && force_redraw) {
        itr = _chunkGlList.end();
    } else
        force_redraw = false;

    if (_chunkGlList.end() == itr) {
        ListCounter cnt;
        if (force_redraw) {
            cnt = itr->second;
            cnt.age = ListCounter::Age_InUse;
        } else {
            cnt.age = ListCounter::Age_JustCreated;
            cnt.displayList = glGenLists(1);
        }

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

void DisplayWidget::drawSpectrogram_borders_directMode( boost::shared_ptr<SpectrogramRenderer> renderer ) {
    glLineWidth(3);
    glColor4f(0,0,0,1);
    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    unsigned sz=10;
    pTransform t = renderer->spectrogram()->transform();//wavelett->getWavelettTransform();
    unsigned f = t->min_hz();
    f = f/sz*sz;
    float l = renderer->spectrogram()->transform()->original_waveform()->length();
    while(f < t->max_hz())
    {
        // float period = start*exp(-ff*steplogsize);
        // f = 1/period = 1/start*exp(ff*steplogsize)
        // start = t->sampleRate/t->minHz/n.width;
        float steplogsize = log(t->max_hz()) - log(t->min_hz());

        float ff = log(f/t->min_hz())/steplogsize;
        if (ff>1)
            break;
        float g=(f/sz == 1)?2:1;
        glLineWidth(g);
    glBegin(GL_LINES);
        glVertex3f(-.015f*g, 0, ff);
        glVertex3f(0.f, 0, ff);
        glVertex3f( l+.015f*g, 0, ff);
        glVertex3f( l, 0, ff);
    glEnd();
        f += sz;
        if(f/sz >= 10) {
            sz*=10;

            glLineWidth(1);
            glPushMatrix();
            glTranslatef(-.03f,0,ff);
            glRotatef(90,0,1,0);
            glRotatef(90,1,0,0);
            glScalef(0.0002f,0.0001f,0.0001f);
            char a[100];
            sprintf(a,"%d", f);
            for (char*c=a;*c!=0; c++)
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
            glPopMatrix();
        }
    }

    for( unsigned tone = (unsigned)ceil(log(20.f)/0.05); true; tone++)
    {
        float steplogsize = log(t->max_hz())-log(t->min_hz());
        float ff = log(exp(tone*.05)/t->min_hz())/steplogsize;
        float ffN = log(exp((tone+1)*.05)/t->min_hz())/steplogsize;
        float ffP = log(exp((tone-1)*.05)/t->min_hz())/steplogsize;
        if (ff>1)
            break;
        bool blackKey = false;
        switch(tone%12) { case 1: case 3: case 6: case 8: case 10: blackKey = true; }
        bool blackKeyP = false;
        switch((tone+11)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyP = true; }
        bool blackKeyN = false;
        switch((tone+1)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyN = true; }
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

    glBegin(blackKey ? GL_QUADS:GL_LINE_LOOP);
        glVertex3f(-.04f -.012f*blackKey, 0, ff+wN);
        glVertex3f(-.07f, 0, ff+wN);
        glVertex3f(-.07f, 0, ff-wP);
        glVertex3f(-.04f -.012f*blackKey, 0, ff-wP);
    glEnd();
        if(tone%12 == 0) {
            glLineWidth(1.f);
            glPushMatrix();
            glTranslatef(-.0515f,0,ff-wP*.7f);
            //glRotatef(90,0,1,0);
            glRotatef(90,1,0,0);
            float s = (wN+wP)*0.01f*.7f;
            glScalef(s*.5f,s,s);
            char a[100];
            sprintf(a,"C%d", tone/12 - 10);
            for (char*c=a;*c!=0; c++)
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
            glPopMatrix();
        }
    }

    unsigned m=0;
    unsigned marker = max(1.f, 20.f/gDisplayWidget->xscale);
    marker = pow(10,ceil(log(marker)/log(10)));

    for (float s=0; s<l;s+=.01f, m++)
    {
        if((m%max((unsigned)1,marker/10))!=0)
            continue;

        float g = (m%marker)==0?2:1;
        glLineWidth(g);
glBegin(GL_LINES);
        glVertex3f(s, 0, -.015f*g);
        glVertex3f(s, 0, 0.f);
        glVertex3f(s, 0, 1+.015f*g);
        glVertex3f(s, 0, 1.f);
glEnd();
        if (0==(m%marker)) {
            glLineWidth(1);
            glPushMatrix();
            glTranslatef(s+.005,0,-.065f);
            glRotatef(90,1,0,0);
            glScalef(0.00045f/gDisplayWidget->xscale,0.0003f,0.0003f);
            char a[100];
            sprintf(a,"%.1f", s);
            for (char*c=a;*c!=0; c++)
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
            glPopMatrix();
        }
    }
    glDepthMask(true);
    glDisable(GL_BLEND);
}

void DisplayWidget::drawSelection() {
    drawSelectionCircle();
}

void DisplayWidget::drawSelectionSquare() {
    float l = _renderer->spectrogram()->transform()->original_waveform()->length();
    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);
    float
            x1 = max(0.f, min(selection[0].x, selection[1].x)),
            z1 = max(0.f, min(selection[0].z, selection[1].z)),
            x2 = min(l, max(selection[0].x, selection[1].x)),
            z2 = min(1.f, max(selection[0].z, selection[1].z));
    float y = 1;


    glBegin(GL_QUADS);
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, y, 1 );
        glVertex3f( x1, y, 1 );
        glVertex3f( x1, y, 0 );

        glVertex3f( x1, y, 0 );
        glVertex3f( x2, y, 0 );
        glVertex3f( x2, y, z1 );
        glVertex3f( x1, y, z1 );

        glVertex3f( x1, y, 1 );
        glVertex3f( x2, y, 1 );
        glVertex3f( x2, y, z2 );
        glVertex3f( x1, y, z2 );

        glVertex3f( l, y, 0 );
        glVertex3f( l, y, 1 );
        glVertex3f( x2, y, 1 );
        glVertex3f( x2, y, 0 );


        if (x1>0) {
            glVertex3f( x1, y, z1 );
            glVertex3f( x1, 0, z1 );
            glVertex3f( x1, 0, z2 );
            glVertex3f( x1, y, z2 );
            glVertex3f( 0, y, 0 );
            glVertex3f( 0, 0, 0 );
            glVertex3f( 0, 0, 1 );
            glVertex3f( 0, y, 1 );
        } else {
            glVertex3f( 0, y, 0 );
            glVertex3f( 0, 0, 0 );
            glVertex3f( 0, 0, z1 );
            glVertex3f( 0, y, z1 );
            glVertex3f( 0, y, z2 );
            glVertex3f( 0, 0, z2 );
            glVertex3f( 0, 0, 1 );
            glVertex3f( 0, y, 1 );
        }

        if (x2<l) {
            glVertex3f( x2, y, z1 );
            glVertex3f( x2, 0, z1 );
            glVertex3f( x2, 0, z2 );
            glVertex3f( x2, y, z2 );
            glVertex3f( l, y, 0 );
            glVertex3f( l, 0, 0 );
            glVertex3f( l, 0, 1 );
            glVertex3f( l, y, 1 );
        } else {
            glVertex3f( l, y, 0 );
            glVertex3f( l, 0, 0 );
            glVertex3f( l, 0, z1 );
            glVertex3f( l, y, z1 );
            glVertex3f( l, y, z2 );
            glVertex3f( l, 0, z2 );
            glVertex3f( l, 0, 1 );
            glVertex3f( l, y, 1 );
        }

        if (z1>0) {
            glVertex3f( x1, y, z1 );
            glVertex3f( x1, 0, z1 );
            glVertex3f( x2, 0, z1 );
            glVertex3f( x2, y, z1 );
            glVertex3f( 0, y, 0 );
            glVertex3f( 0, 0, 0 );
            glVertex3f( l, 0, 0 );
            glVertex3f( l, y, 0 );
        } else {
            glVertex3f( 0, y, 0 );
            glVertex3f( 0, 0, 0 );
            glVertex3f( x1, 0, 0 );
            glVertex3f( x1, y, 0 );
            glVertex3f( x2, y, 0 );
            glVertex3f( x2, 0, 0 );
            glVertex3f( l, 0, 0 );
            glVertex3f( l, y, 0 );
        }

        if (z2<1) {
            glVertex3f( x1, y, z2 );
            glVertex3f( x1, 0, z2 );
            glVertex3f( x2, 0, z2 );
            glVertex3f( x2, y, z2 );
            glVertex3f( 0, y, 1 );
            glVertex3f( 0, 0, 1 );
            glVertex3f( l, 0, 1 );
            glVertex3f( l, y, 1 );
        } else {
            glVertex3f( 0, y, 1 );
            glVertex3f( 0, 0, 1 );
            glVertex3f( x1, 0, 1 );
            glVertex3f( x1, y, 1 );
            glVertex3f( x2, y, 1 );
            glVertex3f( x2, 0, 1 );
            glVertex3f( l, 0, 1 );
            glVertex3f( l, y, 1 );
        }
    glEnd();
    glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
        if (x1>0) {
            glVertex3f( x1, y, z1 );
            glVertex3f( x1, 0, z1 );
            glVertex3f( x1, 0, z2 );
            glVertex3f( x1, y, z2 );
        }

        if (x2<l) {
            glVertex3f( x2, y, z1 );
            glVertex3f( x2, 0, z1 );
            glVertex3f( x2, 0, z2 );
            glVertex3f( x2, y, z2 );
        }

        if (z1>0) {
            glVertex3f( x1, y, z1 );
            glVertex3f( x1, 0, z1 );
            glVertex3f( x2, 0, z1 );
            glVertex3f( x2, y, z1 );
        }

        if (z2<1) {
            glVertex3f( x1, y, z2 );
            glVertex3f( x1, 0, z2 );
            glVertex3f( x2, 0, z2 );
            glVertex3f( x2, y, z2 );
        }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

bool DisplayWidget::insideCircle( float x1, float z1 ) {
    float
            x = selection[0].x,
            z = selection[0].z,
            _rx = selection[1].x,
            _rz = selection[1].z;
    return (x-x1)*(x-x1)/_rx/_rx + (z-z1)*(z-z1)/_rz/_rz < 1;
}

void DisplayWidget::drawSelectionCircle() {
    float
            x = selection[0].x,
            z = selection[0].z,
            _rx = fabs(selection[1].x-selection[0].x),
            _rz = fabs(selection[1].z-selection[0].z);
    float y = 1;

    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);
    glBegin(GL_TRIANGLE_STRIP);
    for (unsigned k=0; k<=360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, 0, s );
        glVertex3f( c, y, s );
    }
    glEnd();

    glLineWidth(3.2f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINE_LOOP);
    for (unsigned k=0; k<360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, y, s );
    }
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
    glDisable(GL_BLEND);
}

void DisplayWidget::drawSelectionCircle2() {
    float l = _renderer->spectrogram()->transform()->original_waveform()->length();
    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    float
            x = selection[0].x,
            z = selection[0].z,
            _rx = fabs(selection[1].x-selection[0].x),
            _rz = fabs(selection[1].z-selection[0].z);
    float y = 1;
    // compute points in each quadrant, upper right
    std::vector<GLvector> pts[4];
    GLvector corner[4];
    corner[0] = GLvector(l,0,1);
    corner[1] = GLvector(0,0,1);
    corner[2] = GLvector(0,0,0);
    corner[3] = GLvector(l,0,0);
    for (unsigned k,j=0; j<4; j++) {
        bool addedLast=false;
        for (k=0; k<=90; k++) {
            float s = z + _rz*sin((k+j*90)*M_PI/180);
            float c = x + _rx*cos((k+j*90)*M_PI/180);
            if (s>0 && s<1 && c>0&&c<l) {
                if (pts[j].empty() && k>0) {
                    if (0==j) pts[j].push_back(GLvector( l, 0, z + _rz*sin(acos((l-x)/_rx))));
                    if (1==j) pts[j].push_back(GLvector( x + _rx*cos(asin((1-z)/_rz)), 0, 1));
                    if (2==j) pts[j].push_back(GLvector( 0, 0, z + _rz*sin(acos((0-x)/_rx))));
                    if (3==j) pts[j].push_back(GLvector( x + _rx*cos(asin((0-z)/_rz)), 0, 0));
                }
                pts[j].push_back(GLvector( c, 0, s));
                addedLast = 90==k;
            }
        }
        if (!addedLast) {
            if (0==j) pts[j].push_back(GLvector( x + _rx*cos(asin((1-z)/_rz)), 0, 1));
            if (1==j) pts[j].push_back(GLvector( 0, 0, z + _rz*sin(acos((0-x)/_rx))));
            if (2==j) pts[j].push_back(GLvector( x + _rx*cos(asin((0-z)/_rz)), 0, 0));
            if (3==j) pts[j].push_back(GLvector( l, 0, z + _rz*sin(acos((l-x)/_rx))));
        }
    }

    for (unsigned j=0; j<4; j++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (unsigned k=0; k<pts[j].size(); k++) {
            glVertex3f( pts[j][k][0], 0, pts[j][k][2] );
            glVertex3f( pts[j][k][0], y, pts[j][k][2] );
        }
        glEnd();
    }


    for (unsigned j=0; j<4; j++) {
        if ( !insideCircle(corner[j][0], corner[j][2]) )
        {
            glBegin(GL_TRIANGLE_FAN);
            GLvector middle1( 0==j?l:2==j?0:corner[j][0], 0, 1==j?1:3==j?0:corner[j][2]);
            GLvector middle2( 3==j?l:1==j?0:corner[j][0], 0, 0==j?1:2==j?0:corner[j][2]);
            if ( !insideCircle(middle1[0], middle1[2]) )
                glVertex3f( middle1[0], y, middle1[2] );
            for (unsigned k=0; k<pts[j].size(); k++) {
                glVertex3f( pts[j][k][0], y, pts[j][k][2] );
            }
            if ( !insideCircle(middle2[0], middle2[2]) )
                glVertex3f( middle2[0], y, middle2[2] );
            glEnd();
        }
    }
    for (unsigned j=0; j<4; j++) {
        bool b1 = insideCircle(corner[j][0], corner[j][2]);
        bool b2 = insideCircle(0==j?l:2==j?0:corner[j][0], 1==j?1:3==j?0:corner[j][2]);
        bool b3 = insideCircle(corner[(j+1)%4][0], corner[(j+1)%4][2]);
        glBegin(GL_TRIANGLE_STRIP);
        if ( b1 )
        {
            glVertex3f( corner[j][0], 0, corner[j][2] );
            glVertex3f( corner[j][0], y, corner[j][2] );
            if ( !b2 && pts[(j+1)%4].size()>0 ) {
                glVertex3f( pts[j].back()[0], 0, pts[j].back()[2] );
                glVertex3f( pts[j].back()[0], y, pts[j].back()[2] );
            }
        }
        if ( b3 ) {
            if ( !b2 && pts[(j+1)%4].size()>1) {
                glVertex3f( pts[(j+1)%4][1][0], 0, pts[(j+1)%4][1][2] );
                glVertex3f( pts[(j+1)%4][1][0], y, pts[(j+1)%4][1][2] );
            }
            glVertex3f( corner[(j+1)%4][0], 0, corner[(j+1)%4][2] );
            glVertex3f( corner[(j+1)%4][0], y, corner[(j+1)%4][2] );
        }
        glEnd();
    }
    glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);

    for (unsigned j=0; j<4; j++) {
        glBegin(GL_LINE_STIPPLE);
        for (unsigned k=0; k<pts[j].size(); k++) {
            glVertex3f( pts[j][k][0], y, pts[j][k][2] );
        }
        glEnd();
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
