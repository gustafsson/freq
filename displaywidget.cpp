#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QTime>
#include <QKeyEvent>

#include <QtGui/QFileDialog>
#include <CudaException.h>
#include <GlException.h>

#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <tvector.h>
#include <math.h>
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif
#include <stdio.h>
#include "signal-audiofile.h"
#include "signal-playback.h"
#include "signal-postsink.h"
#include "signal-microphonerecorder.h"
#include "signal-operation-composite.h"
#include "signal-operation-basic.h"
#include "sawe-csv.h"
#include "signal-writewav.h"

#include <msc_stdc.h>

//#undef max
//#undef min

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

void drawCircleSector(float x, float y, float radius, float start, float end);
void drawRoundRect(float width, float height, float roundness);
void drawRect(float x, float y, float width, float height);
void drawRectRing(int rects, float irad, float orad);

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


DisplayWidget* DisplayWidget::
        gDisplayWidget = 0;

DisplayWidget::
        DisplayWidget(
                Signal::pWorker worker,
                Signal::pSink collection,
                unsigned playback_device,
                std::string selection_filename,
                int timerInterval )
: QGLWidget( ),
  lastKey(0),
  xscale(1),
//  _record_update(false),
  _renderer( new Heightmap::Renderer( dynamic_cast<Heightmap::Collection*>(collection.get()), this )),
  _worker( worker ),
  _collectionCallback( new Signal::WorkerCallback( worker, collection )),
  _postsinkCallback( new Signal::WorkerCallback( worker, Signal::pSink(new Signal::PostSink)) ),
  _selection_filename(selection_filename),
  _playback_device( playback_device ),
  _px(0), _py(0), _pz(-10),
  _rx(91), _ry(180), _rz(0),
  _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
  _prevX(0), _prevY(0), _targetQ(0),
  _selectionActive(true),
  _navigationActive(false),
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
    float l = _worker->source()->length();
    selection[0].x = l*.5f;
    selection[0].y = 0;
    selection[0].z = .85f;
    selection[1].x = l*sqrt(2.0f);
    selection[1].y = 0;
    selection[1].z = 2;
    
    // no selection
    selection[0].x = selection[1].x;
    selection[0].z = selection[1].z;

    yscale = Yscale_LogLinear;
    //timeOut();
    
    if ( timerInterval != 0 )
    {
        startTimer(timerInterval);
    }
        
    if (_rx<0) _rx=0;
    if (_rx>90) { _rx=90; orthoview=1; }
    if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
    
    grabKeyboard();
}

DisplayWidget::
        ~DisplayWidget()
{
    Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
    Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );

    if (r) {
        r->isStopped() ? void() : r->stopRecording();
    }
}

void DisplayWidget::receiveCurrentSelection(int index, bool enabled)
{
    setSelection(index, enabled);
}

void DisplayWidget::receiveFilterRemoval(int index)
{
    removeFilter(index);
}

void DisplayWidget::receiveToggleSelection(bool active)
{
    if(active && _selectionActive != active){
        _navigationActive = false;
        printf("Setting navigation false\n");
        emit setNavigationActive(false);
    }
    _selectionActive = active;
}

void DisplayWidget::receiveToggleNavigation(bool active)
{
    if(active && _navigationActive != active){
        _selectionActive = false;
        printf("Setting selection false\n");
        emit setSelectionActive(false);
    }
    _navigationActive = active;
}

void DisplayWidget::receiveTogglePiano(bool active)
{
    _renderer->draw_piano = active;
    update();
}


void DisplayWidget::receivePlaySound()
{
    TaskTimer tt("Initiating playback of selection.\n");

    Signal::PostSink* postsink = dynamic_cast<Signal::PostSink*>(_postsinkCallback->sink().get());
    postsink->sinks.clear();

    // If no filter, no selection...
    if (!postsink->inverse_cwt.filter)
        return;

    // find range of selection
    Signal::SamplesIntervalDescriptor sid = postsink->inverse_cwt.filter->coveredInterval( _worker->source()->sample_rate() );
    sid -= postsink->inverse_cwt.filter->getZeroSamples( _worker->source()->sample_rate() );
    const Signal::SamplesIntervalDescriptor::Interval i = sid.getInterval(Signal::SamplesIntervalDescriptor::SampleType_MAX, 0);

    if (i.first == i.last)
        return;

    unsigned L = std::min(i.last, _worker->source()->number_of_samples());
    if (L<=i.first)
        return;

    postsink->sinks.push_back(
            Signal::pSink( new Signal::Playback( _playback_device )) );
    postsink->sinks.push_back(
            Signal::pSink( new Signal::WriteWav( _selection_filename )) );

    _postsinkCallback->sink()->add_expected_samples( Signal::SamplesIntervalDescriptor( i.first, L) );
    _worker->todo_list = _postsinkCallback->sink()->expected_samples();

    _worker->todo_list.print(__FUNCTION__);

    update();
}

void DisplayWidget::receiveToggleHz(bool active)
{
    _renderer->draw_hz = active;
    update();
}

void DisplayWidget::receiveAddClearSelection(bool active)
{
    receiveAddSelection(active);

    getFilterOperation()->filter()->enabled = true;

    setWorkerSource();
    update();
}

void DisplayWidget::setWorkerSource( Signal::pSource s ) {
    if (s.get())
        _worker->source( s );

    // Update worker structure
    Signal::FilterOperation* fo = getFilterOperation();

    emit filterChainUpdated(fo->filter());
    emit operationsUpdated( _worker->source() );
}

void DisplayWidget::receiveAddSelection(bool /*active*/)
{
    Signal::PostSink* postsink = dynamic_cast<Signal::PostSink*>(_postsinkCallback->sink().get());
    BOOST_ASSERT( postsink );
    if (!postsink->inverse_cwt.filter)
        return;

    { // If selection is an ellips, remove tfr data inside the ellips
        Tfr::EllipsFilter* ef = dynamic_cast<Tfr::EllipsFilter*>( postsink->inverse_cwt.filter.get() );
        if (ef)
            ef->_save_inside = false;
    }

    Signal::FilterOperation *f;

    Signal::pSource postsink_filter( f = new Signal::FilterOperation( _worker->source(), postsink->inverse_cwt.filter ));
    f->meldFilters();

    { // Test: MoveFilter
     /*   Tfr::pFilter move( new Tfr::MoveFilter( 10 ));
        postsink_filter.reset(f = new Signal::FilterOperation( postsink_filter, move ));
        f->meldFilters();*/
    }

    _renderer->collection()->add_expected_samples( f->filter()->getTouchedSamples(f->sample_rate()) );

    setWorkerSource( postsink_filter );
    update();
}

void DisplayWidget::
        receiveCropSelection()
{
    Signal::Operation *b = getFilterOperation();

    // Find out what to crop based on selection
    unsigned FS = b->sample_rate();
    float radie = fabsf(selection[0].x - selection[1].x);
    unsigned start = std::max(0.f, selection[0].x - radie/sqrtf(2.f)) * FS;
    unsigned end = (selection[0].x + radie/sqrt(2.f)) * FS;

    if (end<=start)
        return;

    // Create OperationRemoveSection to remove that section from the stream
    Signal::pSource remove(new Signal::OperationRemoveSection( b->source(), start, end-start ));

    // Invalidate rendering
    Signal::SamplesIntervalDescriptor sid(start, b->number_of_samples());
    _renderer->collection()->add_expected_samples(sid);

    // Update stream
    b->source(remove);

    setWorkerSource();
    update();
}

void DisplayWidget::
        receiveMoveSelection(bool v)
{
    Signal::Operation *b = getFilterOperation();

    if (true==v) { // Button pressed
        // Remember selection
        sourceSelection[0] = selection[0];
        sourceSelection[1] = selection[1];

    } else { // Button released
		Tfr::pFilter filter(new Tfr::EllipsFilter(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, false ));

		unsigned FS = b->sample_rate();
		int delta = (int)(FS * (selection[0].x - sourceSelection[0].x));

		Signal::pSource moveSelection( new Signal::OperationMoveSelection( 
			b->source(), 
			filter, 
			delta, 
			sourceSelection[0].z - selection[0].z));

        // update stream
        b->source(moveSelection);
		setWorkerSource();


		// Invalidate rendering
		Signal::SamplesIntervalDescriptor sid = Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;
		sid -= filter->getUntouchedSamples( FS );

		Signal::SamplesIntervalDescriptor sid2 = sid;
		if (0<delta) sid2 += delta;
		else         sid2 -= -delta;
		sid |= sid2;

                _renderer->collection()->add_expected_samples(sid);
        update();
    }
}

void DisplayWidget::
        receiveMoveSelectionInTime(bool v)
{
    Signal::Operation *b = getFilterOperation();

    if (true==v) { // Button pressed
        // Remember selection
        sourceSelection[0] = selection[0];
        sourceSelection[1] = selection[1];

    } else { // Button released

        // Create operation to move and merge selection,
        unsigned FS = b->sample_rate();
        float fL = fabsf(sourceSelection[0].x - sourceSelection[1].x);
        if (sourceSelection[0].x < 0)
            fL -= sourceSelection[0].x;
        if (sourceSelection[1].x < 0)
            fL -= sourceSelection[1].x;
        unsigned L = FS * fL;
        if (fL < 0 || 0==L)
            return;
        unsigned oldStart = FS * std::max( 0.f, sourceSelection[0].x );
        unsigned newStart = FS * std::max( 0.f, selection[0].x );
        Signal::pSource moveSelection(new Signal::OperationMove( b->source(),
                                                    oldStart,
                                                    L,
                                                    newStart));

        // Invalidate rendering
        Signal::SamplesIntervalDescriptor sid(oldStart, oldStart+L);
        sid |= Signal::SamplesIntervalDescriptor(newStart, newStart+L);
        _renderer->collection()->add_expected_samples(sid);

        // update stream
        b->source(moveSelection );

        setWorkerSource();
        update();
    }
}

void DisplayWidget::keyPressEvent( QKeyEvent *e )
{
    if (e->isAutoRepeat())
        return;
    
    lastKey = e->key();
    // pTransform t = _renderer->spectrogram()->transform();
    switch (lastKey )
    {
        case ' ':
        {
            receivePlaySound();
            break;
        }
        case 'c': case 'C':
        {
            Signal::FilterOperation* f = getFilterOperation();

            Signal::SamplesIntervalDescriptor sid;


            sid |= f->filter()->getTouchedSamples(f->sample_rate());

            // Remove all topmost filters
            setWorkerSource( f->source() );
            
            update();
            // getFilterOperation will recreate an empty FilterOperation
            setWorkerSource();
            break;
        }
        case 'x': case 'X':
        {
            Signal::pSink s( new Sawe::Csv() );
            s->put( Signal::pBuffer(), _worker->source() );
            break;
        }
        case 'r': case 'R':
        {
            printf("Try recording: ");

            Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
            Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );
            if (r)
            {
                printf("succeded!\n");
                r->isStopped() ? r->startRecording( this ) : r->stopRecording();
            }
            else
            {
                printf("failed!\n");;
            }
            break;
        }
    }
}

void DisplayWidget::put(Signal::pBuffer b )
{    
    float newl = _worker->source()->length();
    newl -= 8*Tfr::CwtSingleton::instance()->wavelet_std_t();
    static float prevl = newl;
    if (_qx >= prevl || prevl == newl) // prevl == newl is true for the first put
        _qx = newl;
    prevl = newl;

    if (b) {
        QMutexLocker l(&_invalidRangeMutex);

        _invalidRange |= b->getInterval();
    }

    update();
}

Signal::FilterOperation* DisplayWidget::getFilterOperation()
{
    Signal::pSource s = _worker->source();
    Signal::FilterOperation* f = dynamic_cast<Signal::FilterOperation*>( s.get() );
    if (0 == f) {
        f = new Signal::FilterOperation(s, Tfr::pFilter());
        s.reset( f );
        _worker->source( s );
    }
    return f;
}

void DisplayWidget::keyReleaseEvent ( QKeyEvent *  )
{
    lastKey = 0;
}

void DisplayWidget::mousePressEvent ( QMouseEvent * e )
{
    /*switch ( e->button() )
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
    }*/
    
    if(_navigationActive) {
        switch ( e->button() )
        {
            case Qt::LeftButton:
                moveButton.press( e->x(), this->height() - e->y() );
                break;

            case Qt::RightButton:
                rotateButton.press( e->x(), this->height() - e->y() );
                break;
            default:
                break;
        }
    }else if(_selectionActive)
    {
        switch ( e->button() )
        {
            case Qt::LeftButton:
                selectionButton.press( e->x(), this->height() - e->y() );
                break;
                
            case Qt::RightButton:
                scaleButton.press( e->x(), this->height() - e->y() );
                break;
            default:
                break;
        }
    }
    
    if(leftButton.isDown() && rightButton.isDown())
        selectionButton.press( e->x(), this->height() - e->y() );
    
    update();
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
            moveButton.release();
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
            scaleButton.release();
            rotateButton.release();
            //printf("RightButton: Release\n");
            break;
            
        default:
            break;
    }
    update();
}

void DisplayWidget::wheelEvent ( QWheelEvent *e )
{
    float ps = 0.0005;
    float rs = 0.08;
    if( e->orientation() == Qt::Horizontal )
    {
    		if(e->modifiers().testFlag(Qt::ShiftModifier))
            xscale *= (1-ps * e->delta());
        else
        		_ry -= rs * e->delta();
    }
    else
    {
		if(e->modifiers().testFlag(Qt::ShiftModifier))
            xscale *= (1-ps * e->delta());
        else
	        _pz *= (1+ps * e->delta());
        //_pz -= ps * e->delta();
        
        //_rx -= ps * e->delta();
    }
    
    update();
}

void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
    float rs = 0.2;
    
    int x = e->x(), y = this->height() - e->y();
    
    
    if ( selectionButton.isDown() )
    {
        GLdouble p[2];
        if (selectionButton.worldPos(x, y, p[0], p[1]))
        {
            if (!selecting) {
                selection[0].x = selection[1].x = selectionStart.x = p[0];
                selection[0].y = selection[1].y = selectionStart.y = 0;
                selection[0].z = selection[1].z = selectionStart.z = p[1];
                selecting = true;
            } else {
                float rt = p[0]-selectionStart.x;
                float rf = p[1]-selectionStart.z;
                selection[0].x = selectionStart.x + .5f*rt;
                selection[0].y = 0;
                selection[0].z = selectionStart.z + .5f*rf;
                selection[1].x = selection[0].x + .5f*sqrtf(2.f)*rt;
                selection[1].y = 0;
                selection[1].z = selection[0].z + .5f*sqrtf(2.f)*rf;

                Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(_postsinkCallback->sink().get());

                BOOST_ASSERT( ps );

                ps->inverse_cwt.filter.reset( new Tfr::EllipsFilter(selection[0].x, selection[0].z, selection[1].x, selection[1].z, true ) );
            }
        }
    } 
    if( rotateButton.isDown() ){
        //Controlling the rotation with the left button.
        _ry += (1-orthoview)*rs * rotateButton.deltaX( x );
        _rx -= rs * rotateButton.deltaY( y );
        if (_rx<0) _rx=0;
        if (_rx>90) { _rx=90; orthoview=1; }
        if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
        
    }
    if( moveButton.isDown() )
    {
        //Controlling the the position with the right button.
        GLvector last, current;
        if( moveButton.worldPos(last[0], last[1]) &&
           moveButton.worldPos(x, y, current[0], current[1]) )
        {
            float l = _worker->source()->length();
            
            _qx -= current[0] - last[0];
            _qz -= current[1] - last[1];
            
            if (_qx<0) _qx=0;
            if (_qz<0) _qz=0;
            if (_qz>8.f/5) _qz=8.f/5;
            if (_qx>l) _qx=l;
        }
    }
    
    
    //Updating the buttons
    leftButton.update( x, y );
    rightButton.update( x, y );
    middleButton.update( x, y );
    selectionButton.update( x, y );
    moveButton.update(x, y);
    rotateButton.update(x, y);
    scaleButton.update(x, y);
    
    update  ();
}

#if 0
void DisplayWidget::timeOut()
{
    leftButton.untouch();
    middleButton.untouch();
    rightButton.untouch();
    selectionButton.untouch();
    rotateButton.untouch();
    moveButton.untouch();
    scaleButton.untouch();
    
    if(selectionButton.isDown() && selectionButton.getHold() == 5)
    {
        receivePlaySound();
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
#endif

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
	_renderRatio = (float)width/(float)height;

    height = height?height:1;
    
    glViewport( 0, 0, (GLint)width, (GLint)height );
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    paintGL();
}


void DisplayWidget::paintGL()
{
    TaskTimer tt(TaskTimer::LogVerbose, __FUNCTION__);
    {
        QMutexLocker l(&_invalidRangeMutex);
        if (!_invalidRange.isEmpty()) {
            _renderer->collection()->add_expected_samples( _invalidRange );
            _invalidRange = Signal::SamplesIntervalDescriptor();
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Set up camera position
    {
        glTranslatef( _px, _py, _pz );

        glRotatef( _rx, 1, 0, 0 );
        glRotatef( fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
        glRotatef( _rz, 0, 0, 1 );

        glScalef(-xscale, 1-.99*orthoview, 5);

        glTranslatef( -_qx, -_qy, -_qz );

        orthoview.TimeStep(.08);
    }

    _renderer->draw();
    _renderer->drawAxes();
    
    static bool computing_inverse = false;
    static bool computing_plot = false;
    bool computing_prev = computing_inverse || computing_plot;
    
    computing_inverse = false;
    computing_plot = false;
    
    if (!_postsinkCallback->sink()->expected_samples().isEmpty())
    {
        computing_inverse = true;
    }
    
    if (0 < this->_renderer->collection()->read_unfinished_count()) {
        computing_plot = true;
    }
    
    if(0) {
        if (computing_inverse) {
            glClearColor(.8f, .8f, .8f, 0.0f);
        } else if (computing_plot) {
            glClearColor(.9f, .9f, .9f, 0.0f);
        } else {
            glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        }
    }
    
    if (computing_prev || computing_inverse || computing_plot)
        update();
    
    drawSelection();
    
    static float computing_rotation = 0.0;
    if (computing_prev || computing_inverse || computing_plot)
    {
        glMatrixMode(GL_PROJECTION);
    	glPushMatrix();
    	glMatrixMode(GL_MODELVIEW);
    	glPushMatrix();
    
    	glMatrixMode(GL_PROJECTION);
    	glLoadIdentity();
    	glOrtho(-1 * _renderRatio, 1 * _renderRatio, 1, -1, -1, 1);
    
		glTranslatef(_renderRatio * 1 -0.15, -0.85, 0);
    	glMatrixMode(GL_MODELVIEW);
    	glLoadIdentity();
    	glScalef(0.5, 0.5, 1);
    
    	glEnable(GL_BLEND);
    	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    	glEnable(GL_DEPTH_TEST);
    	glDepthFunc(GL_LESS);
    	
    	glColor4f(1, 1, 1, 0.5);
    	glPushMatrix();
    	glRotatef(computing_rotation, 0, 0, 1);
    	drawRectRing(15, 0.10, 0.145);
    	glRotatef(-2*computing_rotation, 0, 0, 1);
    	drawRectRing(20, 0.15, 0.2);
    	computing_rotation += 5;
    	glPopMatrix();
    	
    	glColor4f(0, 0, 1, 0.5);
    	drawRoundRect(0.5, 0.5, 0.5);
    	glColor4f(1, 1, 1, 0.5);
    	drawRoundRect(0.55, 0.55, 0.55);
    	
    	glDisable(GL_BLEND);
        //glDisable(GL_DEPTH_TEST);
    
    	glMatrixMode(GL_PROJECTION);
    	glPopMatrix();
    	glMatrixMode(GL_MODELVIEW);
    	glPopMatrix();
    }


    if (_postsinkCallback->sink()->finished()) {
        _postsinkCallback->sink()->reset();
    }

    unsigned center = 0;

    //    if (p && p->isUnderfed() && p->expected_samples_left()) {

    if (!_postsinkCallback->sink()->expected_samples().isEmpty())
    {
        _worker->todo_list = _postsinkCallback->sink()->expected_samples();
        //_worker->todo_list.print("Displaywidget - PostSink");
    }
    else
    {
        center = _worker->source()->sample_rate() * _qx;
        _worker->todo_list = _collectionCallback->sink()->expected_samples();
        //_worker->todo_list.print("Displaywidget - Collection");
    }


    if (_worker->workOne( center ))
        update();

    // CudaException_ThreadSynchronize();
    CudaException_CHECK_ERROR();
    GlException_CHECK_ERROR();
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


void drawCircleSector(float x, float y, float radius, float start, float end)
{
	int numSteps = ((end - start) * radius) * 50.0;
	float step = (end - start) / numSteps;
	
	glBegin(GL_TRIANGLE_FAN);
	glVertex3f( x, y, 0.0f);
    for(int i = 0; i <= numSteps; i++)
    {
    	glVertex3f( x + radius * cos(start + i * step), y - radius * sin(start + i * step), 0.0f);
    }
    glEnd();
}

void drawRect(float x, float y, float width, float height)
{
	glBegin(GL_QUADS);
	glVertex3f(x, y, 0.0f);
	glVertex3f(x + width, y, 0.0f);
	glVertex3f(x + width, y + height, 0.0f);
	glVertex3f(x, y + height, 0.0f);
	glEnd();
}

void drawRectRing(int rects, float irad, float orad)
{
	float height = (irad / rects) * M_PI * 1.2;
	float width = orad - irad;
	float step = 360.0 / rects; 
	
	for(int i = 0; i < rects; i++)
	{
		glPushMatrix();
		glRotatef(step * i, 0, 0, 1);
		drawRect(irad, -height/2, width, height);
		glPopMatrix();
	}
}

void drawRoundRect(float width, float height, float roundness)
{
	roundness = fmax(0.01f, roundness);
	float radius = fmin(width, height) * roundness * 0.5;
	width = width - 2.0 * radius;
	height = height - 2.0 * radius;
	
	drawRect(-width/2.0f, -height/2.0f, width, height);
	drawRect(-(width + 2.0 * radius)/2.0f, -height/2.0f, (width + 2.0 * radius), height);
	drawRect(-width/2.0f, -(height + 2.0 * radius)/2.0f, width, (height + 2.0 * radius));
	
	drawCircleSector(-width/2.0f, -height/2.0f, radius, M_PI/2.0f, M_PI);
	drawCircleSector(width/2.0f, -height/2.0f, radius, 0, M_PI/2.0f);
	drawCircleSector(width/2.0f, height/2.0f, radius, -M_PI/2.0f, 0);
	drawCircleSector(-width/2.0f, height/2.0f, radius, -M_PI, -M_PI/2.0f);
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


void DisplayWidget::drawWaveform(Signal::pSource /*waveform*/)
{
    //static pWaveform_chunk chunk = waveform->getChunk( 0, waveform->number_of_samples(), 0, Waveform_chunk::Only_Real );
    // TODO draw waveform
    /*Signal::pBuffer chunk = waveform.get()->getChunkBehind();
    draw_glList<Signal::Buffer>( chunk, DisplayWidget::drawWaveform_chunk_directMode, chunk->modified );
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
    */
}

void DisplayWidget::drawWaveform_chunk_directMode( Signal::pBuffer chunk)
{
    TaskTimer tt(__FUNCTION__);
    cudaExtent n = chunk->waveform_data->getNumberOfElements();
    const float* data = chunk->waveform_data->getCpuMemory();
    
    n.height = 1;
    float ifs = 1./chunk->sample_rate; // step per sample
    /*    float max = 1e-6;
     //for (unsigned c=0; c<n.height; c++)
     {
     unsigned c=0;
     for (unsigned t=0; t<n.width; t++)
     if (fabsf(data[t + c*n.width])>max)
     max = fabsf(data[t + c*n.width]);
     }
     float s = 1/max;
     */
	float s = 1;
    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
    unsigned c=0;
    //    for (unsigned c=0; c<n.height; c++)
    {
        glBegin(GL_TRIANGLE_STRIP);
        //glBegin(GL_POINTS);
        for (unsigned t=0; t<n.width; t+=std::max( (size_t)1 , (n.width/2000) )) {
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

/**
 draw_glList renders 'chunk' by passing it as argument to 'renderFunction' and caches the results in an OpenGL display list.
 When draw_glList is called again with the same 'chunk' it will not call 'renderFunction' but instead draw the previously cached results.
 If 'force_redraw' is set to true, 'renderFunction' will be called again to replace the old cache. 
 */
template<typename RenderData>
void DisplayWidget::draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw )
{
	// do a cache lookup
    std::map<void*, ListCounter>::iterator itr = _chunkGlList.find(chunk.get());
    
	if (_chunkGlList.end() == itr && force_redraw) {
		force_redraw = false;
	}
    
	// cache miss or force_redraw
    if (_chunkGlList.end() == itr || force_redraw) {
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
		// render cache
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

/*void DisplayWidget::drawSpectrogram_borders_directMode( boost::shared_ptr<SpectrogramRenderer> renderer ) {
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
    marker = pow(10.f,ceil(log((float)marker)/log(10.f)));
    
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
*/

void DisplayWidget::setSelection(int index, bool enabled)
{
    Signal::FilterOperation* f = getFilterOperation();
    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>( f->filter().get());
    if (0 == c || (index < 0 || index>=(int)c->size()))
        return;
    
    printf("####Current selection: %d\n", index);
    Tfr::FilterChain::iterator i = c->begin();
    std::advance(i, index);
    Tfr::EllipsFilter *e = dynamic_cast<Tfr::EllipsFilter*>(i->get());
    if (e)
    {
        selection[0].x = e->_t1;
        selection[0].z = e->_f1;
        selection[1].x = e->_t2;
        selection[1].z = e->_f2;

        Signal::PostSink* postsink = dynamic_cast<Signal::PostSink*>(_postsinkCallback->sink().get());
        BOOST_ASSERT( postsink );
        postsink->inverse_cwt.filter.reset( new Tfr::EllipsFilter(*e ));

        if(e->enabled != enabled) {
            e->enabled = enabled;

            _renderer->collection()->add_expected_samples( e->getTouchedSamples(f->sample_rate()) );
        }
    }
    
    update();
}

void DisplayWidget::removeFilter(int index){
    Signal::FilterOperation* f = getFilterOperation();
    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>( f->filter().get());

    if (0==c || index < 0) return;
    
    printf("####Removing filter: %d\n", index);
    
    Tfr::FilterChain::iterator i = c->begin();
    std::advance(i, index);
    Tfr::EllipsFilter *e = dynamic_cast<Tfr::EllipsFilter*>(i->get());
    if (e)
    {
        c->erase(i);

        _renderer->collection()->add_expected_samples( e->getTouchedSamples(f->sample_rate()) );
    }
    update();
    setWorkerSource();
}

void DisplayWidget::
        drawSelection()
{
    drawSelectionCircle();

    static std::vector<unsigned> playback_itr;
    static boost::posix_time::ptime myClock = boost::posix_time::microsec_clock::local_time();

    if (0==_postsinkCallback) {
        playback_itr.clear();
        return;
    }

    // Draw playback marker
    // Find Signal::Playback* instance
    Signal::Playback* pb = 0;
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>( _postsinkCallback->sink().get() );

    BOOST_ASSERT( ps );
    BOOST_FOREACH( Signal::pSink s, ps->sinks )
    {
        if ( 0 != (pb = dynamic_cast<Signal::Playback*>( s.get() )))
            break;
    }

    // No playback instance
    if (!pb) return;

    // Playback has stopped
    if (pb->isStopped()) {
        playback_itr.clear();
        return;
    }
    /*
    myClock = copyClock;

    if (myClock.isNull()) {
        myClock = QTime::currentTime();
        base_itr = pb->playback_itr();
        prev_itr = pb->playback_itr();
    }
*/
    unsigned this_itr = pb->playback_itr();
    if (playback_itr.empty() || this_itr!=playback_itr.back()) {
        playback_itr.push_back( this_itr );
        if (playback_itr.size()>4) playback_itr.erase(playback_itr.begin());
        myClock = boost::posix_time::microsec_clock::local_time();
    }
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - myClock;
    float dt = d.total_milliseconds()*0.001f;
    float y = 1;
    //float t = (/*b->sample_offset + */base_itr) / (float)pb->sample_rate + dt - pb->outputLatency();
    //float t = b->sample_offset / (float)b->sample_rate + pb->time();
    float t = playback_itr[0]/(float)pb->sample_rate() + dt - .5*pb->outputLatency();
    glEnable(GL_BLEND);
    glDepthMask(false);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    float
        x = selection[0].x,
        z = selection[0].z,
        _rx = selection[1].x-selection[0].x,
        _rz = selection[1].z-selection[0].z,
        z1 = z-sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz,
        z2 = z+sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz;


    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();

    glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

#if (defined (_MSCVER) || defined (_MSC_VER))
	Sleep( 10 );
#else
	usleep( 10000 );
#endif

    update();
}

void DisplayWidget::drawSelectionSquare() {
    float l = _worker->source()->length();
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
    float l = _worker->source()->length();
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
