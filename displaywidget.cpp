#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QTime>
#include <QKeyEvent>
#include <QToolTip>
#include <QHBoxLayout>

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
#include "sawe-hdf5.h"
#include "sawe-matlabfilter.h"
#include "sawe-matlaboperation.h"
#include "signal-writewav.h"
#include "sawe-brushtool.h"
#include "sawe-movetool.h"
#include "sawe-selectiontool.h"

#include <msc_stdc.h>
#include <CudaProperties.h>

//#undef max
//#undef min

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

#define SELECTIONTOOL 0
#define NAVIGATIONTOOL 1
#define BRUSHTOOL 2

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

bool MouseControl::worldPos(GLdouble &ox, GLdouble &oy, float scale)
{
    return worldPos(this->lastx, this->lasty, ox, oy, scale);
}
bool MouseControl::worldPos(GLdouble &ox, GLdouble &oy, float scale, DisplayWidget *dw)
{
    return dw->worldPos(this->lastx, this->lasty, ox, oy, scale);
}
bool MouseControl::worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy, float scale)
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

    float minAngle = 3;
    if( s < 0 || world_coord[0][1]-world_coord[1][1] < scale*sin(minAngle *(M_PI/180)) * (world_coord[0]-world_coord[1]).length() )
        return false;
    
    return test[0] && test[1];
}

bool MouseControl::spacePos(GLdouble &out_x, GLdouble &out_y)
{
    return spacePos(this->lastx, this->lasty, out_x, out_y);
}

bool MouseControl::spacePos(GLdouble in_x, GLdouble in_y, GLdouble &out_x, GLdouble &out_y)
{
    bool test;
    GLvector win_coord, world_coord;

    win_coord = GLvector(in_x, in_y, 0.1);

    world_coord = gluUnProject<GLdouble>(win_coord, &test);
    out_x = world_coord[0];
    out_y = world_coord[2];
    return test;
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


DisplayWidget::
        DisplayWidget(
                Signal::pWorker worker,
                Signal::pSink collection )
: QGLWidget( ),
  lastKey(0),
  orthoview(1),
  xscale(1),
//  _record_update(false),
  _renderer( new Heightmap::Renderer( dynamic_cast<Heightmap::Collection*>(collection.get()), this )),
  _worker( worker ),
  _collectionCallback( new Signal::WorkerCallback( worker, collection )),
  _postsinkCallback( new Signal::WorkerCallback( worker, Signal::pSink(new Signal::PostSink)) ),
  _work_timer( new TaskTimer("Benchmarking first work")),
  _follow_play_marker( false ),
  _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
  _px(0), _py(0), _pz(-10),
  _rx(91), _ry(180), _rz(0),
  _playbackMarker(-1),
  _prevX(0), _prevY(0), _targetQ(0),
  _selectionActive(true),
  _navigationActive(false),
  _infoToolActive(false),
  _enqueueGcDisplayList( false ),
  selecting(false)
{
#ifdef _WIN32
    int c=1;
    char* dum="dum\0";
    glutInit(&c,&dum);
#else
#ifndef __APPLE__
    static int c=0;
    if (0==c)
        glutInit(&c,0),
        c = 1;
#endif
#endif
	
    float l = _worker->source()->length();
    _prevLimit = l;
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
            
    receiveSetTimeFrequencyResolution( 50 );

    if (_rx<0) _rx=0;
    if (_rx>90) { _rx=90; orthoview=1; }
    if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
    
    toolz = 0;
    _toolList.push_back(new Sawe::SelectionTool(this));
    _toolList.push_back(new Sawe::NavigationTool(this));
    _toolList.push_back(new Sawe::BrushTool(this));
    
    setTool(1);
	//toolz->setParent(this);
	//addTool(new Sawe::BrushTool(this));
	
    //grabKeyboard();
}

DisplayWidget::
        ~DisplayWidget()
{
    TaskTimer tt("~DisplayWidget");
    _worker->quit();

    Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
    Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );

    if (r) {
        r->isStopped() ? void() : r->stopRecording();
    }
}

void DisplayWidget::setTool(unsigned int tool)
{
    printf("setTool:\n");
    if(tool >= _toolList.size())
        return;
        
    printf(" -toolNum: %d\n", tool);
    
    if(toolz != 0)
        toolz->setParent(0);
    toolz = _toolList[tool];
    toolz->setParent(0);
        
	QVBoxLayout *verticalLayout = new QVBoxLayout();
	verticalLayout->addWidget(toolz);
	verticalLayout->setContentsMargins(0, 0, 0, 0);
	
	if(layout() != 0)
        delete layout();
	setLayout(verticalLayout);
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
    if (active) {
        receiveToggleNavigation( false );
        receiveToggleInfoTool( false );
        setTool(SELECTIONTOOL);
    } else if(!active) {
        emit setSelectionActive( false );
    }
        
}

void DisplayWidget::receiveToggleNavigation(bool active)
{
    if (active) {
        receiveToggleInfoTool( false );
        receiveToggleSelection( false );
        setTool(NAVIGATIONTOOL);
    } else if(!active) {
        emit setNavigationActive( false );
    }
        
}

void DisplayWidget::receiveToggleInfoTool(bool active)
{
    if (active) {
        receiveToggleNavigation( false );
        receiveToggleSelection( false );
    } else if(!active) {
        emit setInfoToolActive( false );
    }

    _infoToolActive = active;
}

void DisplayWidget::receiveTogglePiano(bool active)
{
    _renderer->draw_piano = active;
    update();
}

void DisplayWidget::
        receiveSetRainbowColors()
{
    _renderer->color_mode = Heightmap::Renderer::ColorMode_Rainbow;
    update();
}

void DisplayWidget::
        receiveSetGrayscaleColors()
{
    _renderer->color_mode = Heightmap::Renderer::ColorMode_Grayscale;
    update();
}

void DisplayWidget::
        receiveSetHeightlines( bool value )
{
    _renderer->draw_height_lines = value;
    update();
}

void DisplayWidget::
        receiveSetYScale( int value )
{
    float f = value / 50.f - 1.f;
    _renderer->y_scale = exp( 4.f*f*f * (f>0?1:-1));
    update();
}

void DisplayWidget::
        receiveSetTimeFrequencyResolution( int value )
{
    unsigned FS = _worker->source()->sample_rate();

    Tfr::pCwt c = Tfr::CwtSingleton::instance();
    c->tf_resolution( exp( 4*(value / 50.f - 1.f)) );

    float std_t = c->morlet_std_t(0, FS);
    c->wavelet_std_t( 1.5f * std_t ); // One standard deviation is not enough, but heavy. Two standard deviations are even more heavy.

    _renderer->collection()->add_expected_samples( Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL );
    update();
}

void DisplayWidget::receivePlaySound()
{
    TaskTimer tt("Initiating playback of selection");

    Signal::PostSink* postsink = getPostSink();

    if (!postsink->filter()) {
        tt.info("No filter, no selection");
        return; // If no filter, no selection...
    }

    // TODO define selections by a selection structure. Currently selections
    // are defined from the first sampels that is non-zero affected by a
    // filter, to the last non-zero affected sample.

    if (postsink->sinks().empty())
    {
        std::vector<Signal::pSink> sinks;
        sinks.push_back( Signal::pSink( new Signal::Playback( playback_device )) );
        sinks.push_back( Signal::pSink( new Signal::WriteWav( selection_filename )) );
        postsink->sinks( sinks );
        postsink->filter( postsink->filter(), _worker->source() );
    }

    postsink->onFinished();

    _worker->todo_list( postsink->expected_samples());
    _worker->todo_list().print(__FUNCTION__);

    // Work as slow as possible on the first few chunks and accelerate.
    // This makes signal::Playback compute better estimates on how fast
    // the computations can be expected to finish.
    _worker->samples_per_chunk_hint(1);

    update();
}

void DisplayWidget::receiveFollowPlayMarker( bool v )
{
    _follow_play_marker = v;
}

void DisplayWidget::receiveToggleHz(bool active)
{
    _renderer->draw_hz = active;
    update();
}

void DisplayWidget::receiveAddSelection(bool active)
{
    Signal::PostSink* postSink = getPostSink();
    if (!postSink->filter())
        return;

    postSink->filter()->enabled = false;

    receiveAddClearSelection(active);

    setWorkerSource();
    update();
}
	
bool DisplayWidget::isRecordSource()
{
    Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
    Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );
	return r != 0;
}

void DisplayWidget::receiveRecord(bool active)
{
    Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
    Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );

    if (r)
    {
        TaskTimer tt("%s recording", active?"Starting":"Stoping");
        if (active == r->isStopped())
        {
                r->isStopped() ? r->startRecording( this ) : r->stopRecording();
        }
    }
    else
    {
        TaskTimer tt("receiveRecord was called without a MicrophoneRecorder source");
    }
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


void DisplayWidget::
        setTimeline( Signal::pSink timelineWidget )
{
    _timeline = timelineWidget;
}

bool DisplayWidget::worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy, float scale)
{
    GLdouble s;
    bool test[2];
    GLvector win_coord, world_coord[2];
    
    win_coord = GLvector(x, y, 0.1);
    
    world_coord[0] = gluUnProject<GLdouble>(win_coord, modelMatrix, projectionMatrix, viewportMatrix, &test[0]);
    //printf("CamPos1: %f: %f: %f\n", world_coord[0][0], world_coord[0][1], world_coord[0][2]);
    
    win_coord[2] = 0.6;
    world_coord[1] = gluUnProject<GLdouble>(win_coord, modelMatrix, projectionMatrix, viewportMatrix, &test[1]);
    //printf("CamPos2: %f: %f: %f\n", world_coord[1][0], world_coord[1][1], world_coord[1][2]);
    
    s = (-world_coord[0][1]/(world_coord[1][1]-world_coord[0][1]));
    
    ox = world_coord[0][0] + s * (world_coord[1][0]-world_coord[0][0]);
    oy = world_coord[0][2] + s * (world_coord[1][2]-world_coord[0][2]);

    float minAngle = 3;
    if( s < 0 || world_coord[0][1]-world_coord[1][1] < scale*sin(minAngle *(M_PI/180)) * (world_coord[0]-world_coord[1]).length() )
        return false;
    
    return test[0] && test[1];
}

void DisplayWidget::
        setPosition( float time, float f )
{
    _qx = time;
    _qz = f;

    float l = _worker->source()->length();

    if (_qx<0) _qx=0;
    if (_qz<0) _qz=0;
    if (_qz>1) _qz=1;
    if (_qx>l) _qx=l;

    worker()->requested_fps(30);
    update();
}

void DisplayWidget::receiveAddClearSelection(bool /*active*/)
{
    getPostSink();
    Signal::PostSink* postsink = getPostSink();

    if (!postsink->filter())
        return;

    { // If selection is an ellips, remove tfr data inside the ellips
        Tfr::EllipsFilter* ef = dynamic_cast<Tfr::EllipsFilter*>( postsink->filter().get() );
        if (ef)
            ef->_save_inside = false;
    }

    Signal::FilterOperation *f;

    Signal::pSource postsink_filter( f = new Signal::FilterOperation( _worker->source(), postsink->filter() ));
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
    unsigned start = std::max(0.f, selection[0].x - radie) * FS;
    unsigned end = (selection[0].x + radie) * FS;

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
        Tfr::pFilter filter(new Tfr::EllipsFilter(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, true ));

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

void DisplayWidget::
        receiveMatlabOperation(bool)
{
    Signal::Operation *b = getFilterOperation();

    Signal::pSource read = b->source();
    if (_matlaboperation)
        read = dynamic_cast<Signal::Operation*>(_matlaboperation.get())->source();

    _matlaboperation.reset( new Sawe::MatlabOperation( read, "matlaboperation") );
    b->source( _matlaboperation );
    _worker->start();
    setWorkerSource();
    update();
    _renderer->collection()->add_expected_samples(Signal::SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL);
}

void DisplayWidget::
        receiveMatlabFilter(bool)
{
    Signal::FilterOperation * b = getFilterOperation();

    Signal::pSource read = b->source();
    if (_matlabfilter)
        read = dynamic_cast<Signal::Operation*>(_matlabfilter.get())->source();

    switch(1) {
    case 1: // Everywhere
        {
            Tfr::pFilter f( new Sawe::MatlabFilter( "matlabfilter" ));
            _matlabfilter.reset( new Signal::FilterOperation( read, f));
            b->source( _matlabfilter );
            _worker->start();
        break;
        }
    case 2: // Only inside selection
        {
        Tfr::pFilter f( new Sawe::MatlabFilter( "matlabfilter" ));
        Signal::pSource s( new Signal::FilterOperation( read, f));

        Tfr::EllipsFilter* e = dynamic_cast<Tfr::EllipsFilter*>(getPostSink()->filter().get());
        if (e)
            e->_save_inside = true;

        _matlabfilter.reset( new Signal::FilterOperation( s, getPostSink()->filter()));
        b->source( _matlabfilter );
        break;
        }
    }


    b->meldFilters();
    _renderer->collection()->add_expected_samples(b->filter()->getTouchedSamples( b->sample_rate()));

    setWorkerSource();
    update();
}

void DisplayWidget::
        receiveTonalizeFilter(bool)
{
    Tfr::pFilter f( new Tfr::TonalizeFilter());
    Signal::FilterOperation *m;
    Signal::pSource tonalize( m = new Signal::FilterOperation( _worker->source(), f));
    m->meldFilters();
    setWorkerSource(tonalize);

    _renderer->collection()->add_expected_samples( f->getTouchedSamples(tonalize->sample_rate()) );

    update();
}

void DisplayWidget::
        receiveReassignFilter(bool)
{
    Tfr::pFilter f( new Tfr::ReassignFilter());
    Signal::FilterOperation *m;
    Signal::pSource reassign( m = new Signal::FilterOperation( _worker->source(), f));
    m->meldFilters();
    setWorkerSource(reassign);

    _renderer->collection()->add_expected_samples( f->getTouchedSamples(reassign->sample_rate()) );

    update();
}

/*void DisplayWidget::keyPressEvent( QKeyEvent *e )
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
    }
}
*/

void DisplayWidget::put( Signal::pBuffer b, Signal::pSource )
{
	QMutexLocker l(&_invalidRangeMutex);
    if (b) {

        _invalidRange |= b->getInterval();
    }
	
	// This causes a crash in Mac OS
    //update();
}

void DisplayWidget::add_expected_samples( const Signal::SamplesIntervalDescriptor& )
{
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

Signal::PostSink* DisplayWidget::getPostSink()
{
    Signal::PostSink* postsink = dynamic_cast<Signal::PostSink*>(_postsinkCallback->sink().get());
    BOOST_ASSERT( postsink );
    return postsink;
}

/*void DisplayWidget::keyReleaseEvent ( QKeyEvent *  )
{
    lastKey = 0;
}*/

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
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
            moveButton.press( e->x(), this->height() - e->y() );

        if( (e->button() & Qt::RightButton) == Qt::RightButton)
            rotateButton.press( e->x(), this->height() - e->y() );

    } else if(_selectionActive)
    {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
            selectionButton.press( e->x(), this->height() - e->y() );

        if( (e->button() & Qt::RightButton) == Qt::RightButton)
            scaleButton.press( e->x(), this->height() - e->y() );

    } else if(_infoToolActive) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton) {
            infoToolButton.press( e->x(), this->height() - e->y() );
            mouseMoveEvent( e );
        }
    }
    
//    if(leftButton.isDown() && rightButton.isDown())
//        selectionButton.press( e->x(), this->height() - e->y() );
    
    update();
    _prevX = e->x(),
    _prevY = e->y();
}

void DisplayWidget::mouseReleaseEvent ( QMouseEvent * e )
{
    switch ( e->button() )
    {
        case Qt::LeftButton:
            //leftButton.release();
            selectionButton.release();
            moveButton.release();
            infoToolButton.release();
            //printf("LeftButton: Release\n");
            selecting = false;
            break;
            
        case Qt::MidButton:
            //middleButton.release();
            //printf("MidButton: Release\n");
            break;
            
        case Qt::RightButton:
            //rightButton.release();
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

        if (_pz<-40) _pz = -40;
        if (_pz>-.4) _pz = -.4;
        //_pz -= ps * e->delta();
        
        //_rx -= ps * e->delta();
    }
    
    update();
}

void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
    makeCurrent();

    float rs = 0.2;
    
    int x = e->x(), y = this->height() - e->y();
//    TaskTimer tt("moving");
    
    if ( selectionButton.isDown() )
    {
        GLdouble p[2];
        if (selectionButton.worldPos(x, y, p[0], p[1], xscale))
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

                getPostSink()->filter( Tfr::pFilter( new Tfr::EllipsFilter(selection[0].x, selection[0].z, selection[1].x, selection[1].z, true )), _worker->source() );
            }
        }
    }
    if (scaleButton.isDown()) {
        // TODO scale selection
    }
    if( rotateButton.isDown() ){
        //Controlling the rotation with the left button.
        _ry += (1-orthoview)*rs * rotateButton.deltaX( x );
        _rx -= rs * rotateButton.deltaY( y );
        if (_rx<10) _rx=10;
        if (_rx>90) { _rx=90; orthoview=1; }
        if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
        
    }

    if( moveButton.isDown() )
    {
        //Controlling the position with the right button.
        GLvector last, current;
        if( moveButton.worldPos(last[0], last[1], xscale) &&
            moveButton.worldPos(x, y, current[0], current[1], xscale) )
        {
            float l = _worker->source()->length();
            
            _qx -= current[0] - last[0];
            _qz -= current[1] - last[1];
            
            if (_qx<0) _qx=0;
            if (_qz<0) _qz=0;
            if (_qz>1) _qz=1;
            if (_qx>l) _qx=l;
        }
    }

    if (infoToolButton.isDown())
    {
        GLvector current;
        if( infoToolButton.worldPos(x, y, current[0], current[1], xscale) )
        {
            const Tfr::pCwt c = Tfr::CwtSingleton::instance();
            unsigned FS = _worker->source()->sample_rate();
            float t = ((unsigned)(current[0]*FS+.5f))/(float)FS;
            current[1] = ((unsigned)(current[1]*c->nScales(FS)+.5f))/(float)c->nScales(FS);
            float f = c->compute_frequency( current[1], FS );
            float std_t = c->morlet_std_t(current[1], FS);
            float std_f = c->morlet_std_f(current[1], FS);

            stringstream ss;
            ss << setiosflags(ios::fixed)
               << "Time: " << setprecision(3) << t << " s" << endl
               << "Frequency: " << setprecision(1) << f << " Hz" << endl
               << "Standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz";
            QToolTip::showText( e->globalPos(), QString::fromLocal8Bit("..."), this ); // Force tooltip to change position even if the text is the same as in previous tooltip
            QToolTip::showText( e->globalPos(), QString::fromLocal8Bit(ss.str().c_str()), this );
        }
    }
    
    //Updating the buttons
    //leftButton.update( x, y );
    //rightButton.update( x, y );
    //middleButton.update( x, y );
    selectionButton.update( x, y );
    moveButton.update(x, y);
    rotateButton.update(x, y);
    scaleButton.update(x, y);
    infoToolButton.update(x, y);
    
    worker()->requested_fps(30);
    update();
}

/*void DisplayWidget::timeOut()
{
    update();
}*/

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

static void printQGLFormat(const QGLFormat& f, std::string title)
{
    TaskTimer tt("QGLFormat %s", title.c_str());
    tt.info("accum=%d",f.accum());
    tt.info("accumBufferSize=%d",f.accumBufferSize());
    tt.info("alpha=%d",f.alpha());
    tt.info("alphaBufferSize=%d",f.alphaBufferSize());
    tt.info("blueBufferSize=%d",f.blueBufferSize());
    tt.info("depth=%d",f.depth());
    tt.info("depthBufferSize=%d",f.depthBufferSize());
    tt.info("directRendering=%d",f.directRendering());
    tt.info("doubleBuffer=%d",f.doubleBuffer());
    tt.info("greenBufferSize=%d",f.greenBufferSize());
    tt.info("hasOverlay=%d",f.hasOverlay());
    tt.info("redBufferSize=%d",f.redBufferSize());
    tt.info("rgba=%d",f.rgba());
    tt.info("sampleBuffers=%d",f.sampleBuffers());
    tt.info("samples=%d",f.samples());
    tt.info("stencil=%d",f.stencil());
    tt.info("stencilBufferSize=%d",f.stencilBufferSize());
    tt.info("stereo=%d",f.stereo());
    tt.info("swapInterval=%d",f.swapInterval());
    tt.info("");
    tt.info("hasOpenGL=%d",f.hasOpenGL());
    tt.info("hasOpenGLOverlays=%d",f.hasOpenGLOverlays());
    QGLFormat::OpenGLVersionFlags flag = f.openGLVersionFlags();
    tt.info("OpenGL_Version_None=%d", QGLFormat::OpenGL_Version_None == flag);
    tt.info("OpenGL_Version_1_1=%d", QGLFormat::OpenGL_Version_1_1 & flag);
    tt.info("OpenGL_Version_1_2=%d", QGLFormat::OpenGL_Version_1_2 & flag);
    tt.info("OpenGL_Version_1_3=%d", QGLFormat::OpenGL_Version_1_3 & flag);
    tt.info("OpenGL_Version_1_4=%d", QGLFormat::OpenGL_Version_1_4 & flag);
    tt.info("OpenGL_Version_1_5=%d", QGLFormat::OpenGL_Version_1_5 & flag);
    tt.info("OpenGL_Version_2_0=%d", QGLFormat::OpenGL_Version_2_0 & flag);
    tt.info("OpenGL_Version_2_1=%d", QGLFormat::OpenGL_Version_2_1 & flag);
    tt.info("OpenGL_Version_3_0=%d", QGLFormat::OpenGL_Version_3_0 & flag);
    tt.info("OpenGL_ES_CommonLite_Version_1_0=%d", QGLFormat::OpenGL_ES_CommonLite_Version_1_0 & flag);
    tt.info("OpenGL_ES_Common_Version_1_0=%d", QGLFormat::OpenGL_ES_Common_Version_1_0 & flag);
    tt.info("OpenGL_ES_CommonLite_Version_1_1=%d", QGLFormat::OpenGL_ES_CommonLite_Version_1_1 & flag);
    tt.info("OpenGL_ES_Common_Version_1_1=%d", QGLFormat::OpenGL_ES_Common_Version_1_1 & flag);
    tt.info("OpenGL_ES_Version_2_0=%d", QGLFormat::OpenGL_ES_Version_2_0 & flag);
}

static void printQGLWidget(const QGLWidget& w, std::string title)
{
    TaskTimer tt("QGLWidget %s", title.c_str());
    tt.info("doubleBuffer=%d", w.doubleBuffer());
    tt.info("isSharing=%d", w.isSharing());
    tt.info("isValid=%d", w.isValid());
    printQGLFormat( w.format(), "");
}

void DisplayWidget::initializeGL()
{
    //printQGLWidget(*this, "this");
    //TaskTimer("autoBufferSwap=%d", autoBufferSwap()).suppressTiming();

    glShadeModel(GL_SMOOTH);
    
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
    }

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
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,10000.0f);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void DisplayWidget::paintGL()
{
    TIME_PAINTGL _render_timer.reset();
    TIME_PAINTGL _render_timer.reset(new TaskTimer("Time since last DisplayWidget::paintGL"));

    static int tryGc = 0;
    try {
        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();

    {   QMutexLocker l(&_invalidRangeMutex); // 0.00 ms
        if (!_invalidRange.isEmpty()) {
            Signal::SamplesIntervalDescriptor blur = _invalidRange;
            unsigned fuzzy = Tfr::CwtSingleton::instance()->wavelet_std_samples(_worker->source()->sample_rate());
            blur += fuzzy;
            _invalidRange |= blur;

            blur = _invalidRange;
            blur -= fuzzy;
            _invalidRange |= blur;

            _renderer->collection()->add_expected_samples( _invalidRange );
            _invalidRange = Signal::SamplesIntervalDescriptor();
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up camera position
    bool followingRecordMarker = false;
    float length = _worker->source()->length();
    {   double limit = std::max(0.f, length - 2*Tfr::CwtSingleton::instance()->wavelet_std_t());
        if (_qx>=_prevLimit) {
            // Snap just before end so that _worker->center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            _qx = std::max(_qx,limit);
            followingRecordMarker = true;
        }
        _prevLimit = limit;

        locatePlaybackMarker();

        setupCamera();
    }

    bool wasWorking = !_worker->todo_list().isEmpty();
    { // Render
        _renderer->collection()->next_frame(); // Discard needed blocks before this row

        _renderer->camera = GLvector(_qx, _qy, _qz);
        _renderer->draw( 1-orthoview ); // 0.6 ms
        _renderer->drawAxes( length ); // 4.7 ms
        drawSelection(); // 0.1 ms
        if(toolz) toolz->render();

        if (wasWorking)
            drawWorking();

        // When drawing displaywidget, always redraw the timeline as the
        // timeline has a marker showing the current render position of
        // displaywidget
        if (_timeline) dynamic_cast<QWidget*>(_timeline.get())->update();
    }

    {   // Find things to work on (ie playback and file output)

        //    if (p && p->isUnderfed() && p->expected_samples_left()) {
        if (!_postsinkCallback->sink()->expected_samples().isEmpty())
        {
            _worker->center = 0;
            _worker->todo_list( _postsinkCallback->sink()->expected_samples() );

            // Request at least 1 fps. Otherwise there is a risk that CUDA
            // will screw up playback by blocking the OS and causing audio
            // starvation.
            worker()->requested_fps(1);

            //_worker->todo_list().print("Displaywidget - PostSink");
        } else {
            _worker->center = _qx;
            _worker->todo_list( _collectionCallback->sink()->expected_samples());
            //_worker->todo_list().print("Displaywidget - Collection");

            if (followingRecordMarker)
                worker()->requested_fps(1);
        }
        Signal::pSource first_source = Signal::Operation::first_source(_worker->source() );
    	Signal::MicrophoneRecorder* r = dynamic_cast<Signal::MicrophoneRecorder*>( first_source.get() );
        if(r != 0 && !(r->isStopped()))
        {
        	wasWorking = true;
        }
    }

    {   // Work
        bool isWorking = !_worker->todo_list().isEmpty();

        if (wasWorking || isWorking) {
            // _worker can be run in one or more separate threads, but if it isn't
            // execute the computations for one chunk
            if (!_worker->isRunning()) {
                _worker->workOne();
                QTimer::singleShot(0, this, SLOT(update())); // this will leave room for others to paint as well, calling 'update' wouldn't
            } else {
                //_worker->todo_list().print("Work to do");
                // Wait a bit while the other thread work
                QTimer::singleShot(200, this, SLOT(update()));

				_worker->checkForErrors();
            }

            if (!_work_timer.get())
                _work_timer.reset( new TaskTimer("Working"));
        } else {
            static unsigned workcount = 0;
            if (_work_timer) {
                _work_timer->info("Finished %u chunks, %g s. Work session #%u", _worker->work_chunks, _worker->work_time, workcount);
                _worker->work_chunks = 0;
                _worker->work_time = 0;
                workcount++;
                _work_timer.reset();
            }
        }
    }

    GlException_CHECK_ERROR();
    CudaException_CHECK_ERROR();

    tryGc = 0;
    } catch (const CudaException &x) {
        TaskTimer tt("DisplayWidget::paintGL CAUGHT CUDAEXCEPTION\n%s", x.what());
        if (2>tryGc) {
        	Heightmap::Collection* c=_renderer->collection();
        	c->reset();
            _renderer.reset();
            _renderer.reset(new Heightmap::Renderer( c, this ));
            tryGc++;
            //cudaThreadExit();
            int count;
            cudaError_t e = cudaGetDeviceCount(&count);
            TaskTimer tt("Number of CUDA devices=%u, error=%s", count, cudaGetErrorString(e));
            // e = cudaThreadExit();
            // tt.info("cudaThreadExit, error=%s", cudaGetErrorString(e));
            //CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
            //e = cudaSetDevice( 1 );
            //tt.info("cudaSetDevice( 1 ), error=%s", cudaGetErrorString(e));
            //e = cudaSetDevice( 0 );
            //tt.info("cudaSetDevice( 0 ), error=%s", cudaGetErrorString(e));
            void *p=0;
            e = cudaMalloc( &p, 10 );
            tt.info("cudaMalloc( 10 ), p=%p, error=%s", p, cudaGetErrorString(e));
            e = cudaFree( p );
            tt.info("cudaFree, error=%s", cudaGetErrorString(e));
            cudaGetLastError();
        }
        else throw;
    } catch (const GlException &x) {
        TaskTimer tt("DisplayWidget::paintGL CAUGHT GLEXCEPTION\n%s", x.what());
        if (0==tryGc) {
            _renderer->collection()->gc();
            tryGc++;
            //cudaThreadExit();
            cudaGetLastError();
        }
        else throw;
    }
    
    glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
    glGetIntegerv(GL_VIEWPORT, viewportMatrix);
}

void DisplayWidget::setupCamera()
{
    glLoadIdentity();
    glTranslatef( _px, _py, _pz );

    glRotatef( _rx, 1, 0, 0 );
    glRotatef( fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( _rz, 0, 0, 1 );

    glScalef(-xscale, 1, 5);

    glTranslatef( -_qx, -_qy, -_qz );

    orthoview.TimeStep(.08);
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
    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
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
    //glDisable(GL_BLEND);
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
    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    
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
    //glDisable(GL_BLEND);
}
*/

void DisplayWidget::setSelection(int index, bool enabled)
{
    printf("Selection happen!\n");
    Signal::FilterOperation* f = getFilterOperation();
    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>( f->filter().get());
    if (0 == c || (index < 0 || index>=(int)c->size()))
        return;
    
    TaskTimer tt("Current selection: %d", index);
    Tfr::FilterChain::iterator i = c->begin();
    std::advance(i, index);
    Tfr::EllipsFilter *e = dynamic_cast<Tfr::EllipsFilter*>(i->get());
    if (e)
    {
        selection[0].x = e->_t1;
        selection[0].z = e->_f1;
        selection[1].x = e->_t2;
        selection[1].z = e->_f2;

        Tfr::EllipsFilter *e2;
        Tfr::pFilter sf( e2 = new Tfr::EllipsFilter(*e ));
        e2->_save_inside = true;
        getPostSink()->filter( sf, _worker->source());
    }

    Tfr::Filter* filter = i->get();
    if(filter->enabled != enabled) {
        filter->enabled = enabled;

        _renderer->collection()->add_expected_samples( filter->getTouchedSamples(f->sample_rate()) );
    }
    
    update();
}

void DisplayWidget::setSelection(MyVector start, MyVector end, bool enabled)
{
    selection[0].x = start.x;
    selection[0].z = start.z;
    selection[1].x = end.x;
    selection[1].z = end.z;
    update();
}

void DisplayWidget::removeFilter(int index){
    Signal::FilterOperation* f = getFilterOperation();
    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>( f->filter().get());

    if (0==c || index < 0) return;
    
    TaskTimer tt("Removing filter: %d", index);
    
    Tfr::FilterChain::iterator i = c->begin();
    std::advance(i, index);
    Tfr::Filter *e = i->get();
    if (e)
    {
        _renderer->collection()->add_expected_samples( e->getTouchedSamples(f->sample_rate()) );

        c->erase(i);
    }

    update();
    setWorkerSource();
}

void DisplayWidget::
        drawWorking()
{
    static float computing_rotation = 0.0;

	glDepthFunc(GL_LEQUAL);
	glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( width(), 0, height(), 0, -1, 1);

    glTranslatef( 30, 30, 0 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(60, 60, 1);

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

    //glDisable(GL_BLEND);
    //glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glDepthFunc(GL_LEQUAL);
}

void DisplayWidget::
        locatePlaybackMarker()
{
    _playbackMarker = -1;

    if (0==_postsinkCallback) {
        return;
    }

    // Draw playback marker
    // Find Signal::Playback* instance
    Signal::Playback* pb = 0;

    BOOST_FOREACH( Signal::pSink s, getPostSink()->sinks() )
    {
        if ( 0 != (pb = dynamic_cast<Signal::Playback*>( s.get() )))
            break;
    }

    // No playback instance
    if (!pb) {
        return;
    }

    // Playback has stopped
    if (pb->isStopped()) {
        return;
    }

    _playbackMarker = pb->time();
    if (_follow_play_marker)
        _qx = _playbackMarker;
}

void DisplayWidget::
        drawPlaybackMarker()
{
    if (0>_playbackMarker)
        return;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    float
        t = _playbackMarker,
        x = selection[0].x,
        y = 1,
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

    //glDisable(GL_BLEND);
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

    QTimer::singleShot(10, this, SLOT(update()));
}

void DisplayWidget::
        drawSelection()
{
    drawSelectionCircle();
    drawPlaybackMarker();
}

void DisplayWidget::drawSelectionSquare() {
    float l = _worker->source()->length();
    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
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
    //glDisable(GL_BLEND);
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
    
    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
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
    //glDisable(GL_BLEND);
}

void DisplayWidget::drawSelectionCircle2() {
    float l = _worker->source()->length();
    glDepthMask(false);
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

    for (unsigned k,j=0; j<4; j++)
	{
        bool addedLast=false;
        for (k=0; k<=90; k++)
		{
            float s = z + _rz*sin((k+j*90)*M_PI/180);
            float c = x + _rx*cos((k+j*90)*M_PI/180);

            if (s>0 && s<1 && c>0&&c<l)
			{
                if (pts[j].empty() && k>0)
				{
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
    //glDisable(GL_BLEND);
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
