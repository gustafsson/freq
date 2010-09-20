#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QTime>
#include <QKeyEvent>
#include <QToolTip>

#include <QtGui/QFileDialog>
#include <CudaException.h>
#include <GlException.h>

#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <tvector.h>
#include <neat_math.h>
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif
#include <stdio.h>
#include "adapters/audiofile.h"
#include "adapters/playback.h"
#include "signal/postsink.h"
#include "adapters/microphonerecorder.h"
#include "signal/operation-composite.h"
#include "signal/operation-basic.h"
#include "adapters/csv.h"
#include "adapters/hdf5.h"
#include "adapters/matlabfilter.h"
#include "adapters/matlaboperation.h"
#include "adapters/writewav.h"
#include "heightmap/blockfilter.h"
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "filters/filters.h"
#include "tfr/cwt.h"
#include "tools/toolfactory.h"

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

namespace Ui {

void drawCircleSector(float x, float y, float radius, float start, float end);
void drawRoundRect(float width, float height, float roundness);
void drawRect(float x, float y, float width, float height);
void drawRectRing(int rects, float irad, float orad);




using namespace std;




DisplayWidget::
        DisplayWidget(
                Sawe::Project* project, QWidget* parent, Tools::RenderModel* model )
: QGLWidget( parent ),
  lastKey(0),
  orthoview(1),
  xscale(1),
//  _record_update(false),
  project( project ),
  _model( model ),
  _work_timer( new TaskTimer("Benchmarking first work")),
  _follow_play_marker( false ),
  _px(0), _py(0), _pz(-10),
  _rx(91), _ry(180), _rz(0),
  _prevX(0), _prevY(0), _targetQ(0),
  _selectionActive(true),
  _navigationActive(false),
  _infoToolActive(false),
  _enqueueGcDisplayList( false ),
  selecting(false)
{
#ifdef _WIN32
    int c=1;
    char* dummy="dummy\0";
    glutInit(&c,&dummy);
#else
    static int c=0;
    if (0==c)
        glutInit(&c,0),
        c = 1;
#endif
    float l = project->worker.source()->length();

    _prevLimit = l;


    yscale = Yscale_LogLinear;
    //timeOut();
            
    receiveSetTimeFrequencyResolution( 50 );

    if (_rx<0) _rx=0;
    if (_rx>90) { _rx=90; orthoview=1; }
    if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
    
    //grabKeyboard();
}

DisplayWidget::
        ~DisplayWidget()
{
    TaskTimer tt("~DisplayWidget");
    project->worker.quit();

    Signal::Operation* first_source =project->worker.source()->root();
    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );

    if (r && !r->isStopped())
         r->stopRecording();
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
    } else if(!active) {
        emit setSelectionActive( false );
    }

    _selectionActive = active;
}

void DisplayWidget::receiveToggleNavigation(bool active)
{
    if (active) {
        receiveToggleInfoTool( false );
        receiveToggleSelection( false );
    } else if(!active) {
        emit setNavigationActive( false );
    }

    _navigationActive = active;
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
    project->tools().render_view.renderer->draw_piano = active;
    update();
}

void DisplayWidget::
        receiveSetRainbowColors()
{
    project->tools().render_view.renderer->color_mode = Heightmap::Renderer::ColorMode_Rainbow;
    update();
}

void DisplayWidget::
        receiveSetGrayscaleColors()
{
    project->tools().render_view.renderer->color_mode = Heightmap::Renderer::ColorMode_Grayscale;
    update();
}

void DisplayWidget::
        receiveSetHeightlines( bool value )
{
    project->tools().render_view.renderer->draw_height_lines = value;
    update();
}

void DisplayWidget::
        receiveSetYScale( int value )
{
    float f = value / 50.f - 1.f;
    project->tools().render_view.renderer->y_scale = exp( 4.f*f*f * (f>0?1:-1));
    update();
}

void DisplayWidget::
        receiveSetTimeFrequencyResolution( int value )
{
    unsigned FS = project->worker.source()->sample_rate();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();
    c.tf_resolution( exp( 4*(value / 50.f - 1.f)) );

    float std_t = c.morlet_std_t(0, FS);

    // One standard deviation is not enough, but heavy. Two standard deviations are even more heavy.
    c.wavelet_std_t( 1.5f * std_t );

    Tfr::Stft& s = Tfr::Stft::Singleton();
    s.set_approximate_chunk_size( c.wavelet_std_t() * FS );

    _model->collection->invalidate_samples( Signal::Intervals::Intervals_ALL );
    update();
}


void DisplayWidget::receivePlaySound()
{
    TaskTimer tt("Initiating playback of selection");

    Signal::PostSink* selection_operations = project->tools().selection_model.getPostSink();

    // TODO define selections by a selection structure. Currently selections
    // are defined from the first sampels that is non-zero affected by a
    // filter, to the last non-zero affected sample.
    if (!selection_operations->filter()) {
        tt.info("No filter, no selection");
        return; // No filter, no selection...
    }

    if (selection_operations->sinks().empty())
    {
        std::vector<Signal::pOperation> sinks;
        sinks.push_back( Signal::pOperation( new Adapters::Playback( playback_device )) );
        sinks.push_back( Signal::pOperation( new Adapters::WriteWav( selection_filename )) );
        selection_operations->sinks( sinks );
    }

    project->worker.todo_list( selection_operations->fetch_invalid_samples() );
    project->worker.todo_list().print(__FUNCTION__);

    // Work 'as slow as possible' on the first few chunks and accelerate.
    // It will soon accelerate to maximum speed.
    // This makes Adapters::Playback compute better estimates on how fast
    // the computations can be expected to finish (and thus start playing
    // sound before the entire sound has been computed).
    project->worker.samples_per_chunk_hint(1);

    update();
}

void DisplayWidget::receiveFollowPlayMarker( bool v )
{
    _follow_play_marker = v;
}

void DisplayWidget::receiveToggleHz(bool active)
{
    project->tools().render_view.renderer->draw_hz = active;
    update();
}

void DisplayWidget::receiveAddSelection(bool active)
{
    Signal::PostSink* postsink = project->tools().selection_model.getPostSink();

    Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(postsink->filter().get());
    if (!f)
        return;

    f->enabled(false);

    receiveAddClearSelection(active);

    setWorkerSource();
    update();
}
	
bool DisplayWidget::isRecordSource()
{
    Signal::Operation* first_source = project->worker.source()->root();
    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );
	return r != 0;
}

void DisplayWidget::receiveRecord(bool active)
{
    Signal::Operation* first_source = project->worker.source()->root();
    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );

    // TODO make connection elsewhere
    //connect(r, SIGNAL(data_available(MicrophoneRecorder*)), SLOT(update()), Qt::UniqueConnection );

    if (r)
    {
        TaskTimer tt("%s recording", active?"Starting":"Stoping");
        if (active == r->isStopped())
        {
            r->isStopped() ? r->startRecording() : r->stopRecording();
        }
    }
    else
    {
        TaskTimer tt("receiveRecord was called without a MicrophoneRecorder source");
    }
    update();
}

void DisplayWidget::receiveSetTransform_Cwt()
{
    Signal::pOperation s = project->tools().render_view.renderer->collection()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(project->tools().render_view.renderer->collection().get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    update();
}

void DisplayWidget::receiveSetTransform_Stft()
{
    Signal::pOperation s = project->tools().render_view.renderer->collection()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::StftToBlock* cwtblock = new Heightmap::StftToBlock(project->tools().render_view.renderer->collection().get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);

    update();
}

void DisplayWidget::receiveSetTransform_Cwt_phase()
{
    Signal::pOperation s = project->tools().render_view.renderer->collection()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(project->tools().render_view.renderer->collection().get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    update();
}

void DisplayWidget::receiveSetTransform_Cwt_reassign()
{
    Signal::pOperation s = project->tools().render_view.renderer->collection()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(project->tools().render_view.renderer->collection().get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    ps->filter( Signal::pOperation(new Filters::Reassign()));

    update();
}

void DisplayWidget::receiveSetTransform_Cwt_ridge()
{
    Signal::pOperation s = project->tools().render_view.renderer->collection()->postsink();
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());

    if (!ps)
        return;

    std::vector<Signal::pOperation> v;
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(project->tools().render_view.renderer->collection().get());
    v.push_back( Signal::pOperation( cwtblock ) );
    ps->sinks(v);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    ps->filter( Signal::pOperation(new Filters::Ridge()));

    update();
}

void DisplayWidget::setWorkerSource( Signal::pOperation s ) {
    if (s.get())
        project->worker.source( s );

    // Update worker structure
    emit operationsUpdated( project->worker.source() );
}




void DisplayWidget::receiveAddClearSelection(bool /*active*/)
{
    Signal::PostSink* postsink = project->tools().selection_model.getPostSink();

    if (!postsink->filter())
        return;

    { // If selection is an ellips, remove tfr data inside the ellips
        Filters::EllipsFilter* ef = dynamic_cast<Filters::EllipsFilter*>( postsink->filter().get() );
        if (ef)
            ef->_save_inside = false;
    }


    Signal::pOperation postsink_filter = postsink->filter();

    { // Test: MoveFilter
        // Tfr::CwtFilter *f;

     /*   Signal::pOperation move( new Tfr::MoveFilter( 10 ));
        postsink_filter.reset(f = new Tfr::CwtFilter( postsink_filter, move ));
        f->meldFilters();*/
        // project->tools().render_view.renderer->collection()->add_expected_samples( f->affected_samples() );
    }

    postsink_filter->source( project->worker.source() );
    setWorkerSource( postsink_filter );
    update();
}

void DisplayWidget::
        receiveCropSelection()
{
    Signal::Operation *b = getCwtFilterHead();

    // Find out what to crop based on selection
    unsigned FS = b->sample_rate();
    MyVector* selection = project->tools().selection_model.selection;
    float radie = fabsf(selection[0].x - selection[1].x);
    unsigned start = std::max(0.f, selection[0].x - radie) * FS;
    unsigned end = (selection[0].x + radie) * FS;

    if (end<=start)
        return;

    // Create OperationRemoveSection to remove that section from the stream
    Signal::pOperation remove(new Signal::OperationRemoveSection( b->source(), start, end-start ));

    // Invalidate rendering
    Signal::Intervals sid(start, b->number_of_samples());
    project->tools().render_view.renderer->collection()->invalidate_samples(sid);

    // Update stream
    b->source(remove);

    setWorkerSource();
    update();
}

void DisplayWidget::
        receiveMoveSelection(bool v)
{
    Signal::Operation *b = getCwtFilterHead();
    MyVector* selection = project->tools().selection_model.selection;

    if (true==v) { // Button pressed
        // Remember selection
        sourceSelection[0] = selection[0];
        sourceSelection[1] = selection[1];

    } else { // Button released
        Signal::pOperation filter(new Filters::EllipsFilter(sourceSelection[0].x, sourceSelection[0].z, sourceSelection[1].x, sourceSelection[1].z, true ));

        unsigned FS = b->sample_rate();
        int delta = (int)(FS * (selection[0].x - sourceSelection[0].x));

        Signal::pOperation moveSelection( new Signal::OperationMoveSelection(
                b->source(),
                filter,
                delta,
                sourceSelection[0].z - selection[0].z));

        // update stream
        b->source(moveSelection);
        setWorkerSource();


        // Invalidate rendering
        Signal::Intervals sid = Signal::Intervals::Intervals_ALL;
        sid -= filter->affected_samples();

        Signal::Intervals sid2 = sid;
        if (0<delta) sid2 += delta;
        else         sid2 -= -delta;
        sid |= sid2;

        project->tools().render_view.renderer->collection()->invalidate_samples(sid);
        update();
    }
}

void DisplayWidget::
        receiveMoveSelectionInTime(bool v)
{
    Signal::Operation *b = getCwtFilterHead();
    MyVector* selection = project->tools().selection_model.selection;

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
        Signal::pOperation moveSelection(new Signal::OperationMove( b->source(),
                                                    oldStart,
                                                    L,
                                                    newStart));

        // Invalidate rendering
        Signal::Intervals sid(oldStart, oldStart+L);
        sid |= Signal::Intervals(newStart, newStart+L);
        project->tools().render_view.renderer->collection()->invalidate_samples(sid);

        // update stream
        b->source(moveSelection );

        setWorkerSource();
        update();
    }
}

void DisplayWidget::
        receiveMatlabOperation(bool)
{
    Signal::Operation *b = getCwtFilterHead();

    Signal::pOperation read = b->source();
    if (_matlaboperation)
        read = dynamic_cast<Signal::Operation*>(_matlaboperation.get())->source();

    _matlaboperation.reset( new Adapters::MatlabOperation( read, "matlaboperation") );
    b->source( _matlaboperation );
    project->worker.start();
    setWorkerSource();
    update();
    project->tools().render_view.renderer->collection()->invalidate_samples(Signal::Intervals::Intervals_ALL);
}

void DisplayWidget::
        receiveMatlabFilter(bool)
{
    Tfr::CwtFilter * b = getCwtFilterHead();

    Signal::pOperation read = b->source();
    if (_matlabfilter)
        read = dynamic_cast<Signal::Operation*>(_matlabfilter.get())->source();

    switch(1) {
    case 1: // Everywhere
        {
            _matlabfilter.reset( new Adapters::MatlabFilter( "matlabfilter" ) );
            _matlabfilter->source( read );
            b->source( _matlabfilter );
            project->worker.start();
        break;
        }
    case 2: // Only inside selection
        {
        Signal::pOperation s( new Adapters::MatlabFilter( "matlabfilter" ));
        s->source( read );

        Signal::PostSink* postsink = project->tools().selection_model.getPostSink();

        Filters::EllipsFilter* e = dynamic_cast<Filters::EllipsFilter*>(postsink->filter().get());
        if (e)
            e->_save_inside = true;

        _matlabfilter = postsink->filter();
        postsink->filter(Signal::pOperation());
        _matlabfilter->source( s );

        b->source( _matlabfilter );
        break;
        }
    }


    project->tools().render_view.renderer->collection()->invalidate_samples(b->affected_samples());

    setWorkerSource();
    update();
}

void DisplayWidget::
        receiveTonalizeFilter(bool)
{
    Signal::pOperation tonalize( new Filters::TonalizeFilter());
    tonalize->source( project->worker.source() );

    setWorkerSource(tonalize);

    project->tools().render_view.renderer->collection()->invalidate_samples( tonalize->affected_samples());

    update();
}

void DisplayWidget::
        receiveReassignFilter(bool)
{
    Signal::pOperation reassign( new Filters::ReassignFilter());
    reassign->source(project->worker.source());
    setWorkerSource(reassign);

    project->tools().render_view.renderer->collection()->invalidate_samples( reassign->affected_samples() );

    update();
}

/*void DisplayWidget::keyPressEvent( QKeyEvent *e )
{
    if (e->isAutoRepeat())
        return;
    
    lastKey = e->key();
    // pTransform t = project->tools().render_view.renderer->spectrogram()->transform();
    switch (lastKey )
    {
        case ' ':
        {
            receivePlaySound();
            break;
        }
        case 'c': case 'C':
        {
            Tfr::CwtFilter* f = getCwtFilter();

            Signal::SamplesIntervalDescriptor sid;


            sid |= f->filter()->getTouchedSamples(f->sample_rate());

            // Remove all topmost filters
            setWorkerSource( f->source() );
            
            update();
            // getCwtFilter will recreate an empty CwtFilter
            setWorkerSource();
            break;
        }
        case 'x': case 'X':
        {
            Signal::pSink s( new Sawe::Csv() );
            s->put( Signal::pBuffer(), project->worker.source() );
            break;
        }
    }
}
*/

/*todo remove
void DisplayWidget::put( Signal::pBuffer b, Signal::pOperation )
{
    QMutexLocker l(&_invalidRangeMutex);
    if (b) {
        _invalidRange |= b->getInterval();
    }
	
    // This causes a crash in Mac OS
    // reason: update is a slot in a QObject whose thread is different from the calling thread (put was called from microphonerecorder)
    // solution was to invokeMethod or declare a signal, connect it and emit the signal
    // http://stackoverflow.com/questions/1144240/qt-how-to-call-slot-from-custom-c-code-running-in-a-different-thread
    // strange that it resulted in a crash instead of an assertion failure on
    // Q_ASSERT(qApp && qApp->thread() == QThread::currentThread());
    // maybe it did, but we didn't see it in the dev environment on Mac?
    //update();
}*/


Heightmap::pCollection DisplayWidget::
        collection()
{
    return project->tools().render_model.collection;
}


Heightmap::pRenderer DisplayWidget::
        renderer()
{
    return project->tools().render_view.renderer;
}


Tfr::CwtFilter* DisplayWidget::getCwtFilterHead()
{
    Signal::pOperation s = project->worker.source();
    Tfr::CwtFilter* f = dynamic_cast<Tfr::CwtFilter*>( s.get() );
    if (0 == f) {
        f = new Tfr::DummyCwtFilter();
        f->source( s );
        project->worker.source( Signal::pOperation(f) );
    }
    return f;
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
        if (_pz>-.1) _pz = -.1;
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
        MyVector* selection = project->tools().selection_model.selection;
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

                Signal::PostSink* postsink = project->tools().selection_model.getPostSink();

                Signal::pOperation newFilter( new Filters::EllipsFilter(selection[0].x, selection[0].z, selection[1].x, selection[1].z, true ));
                newFilter->source(project->worker.source());
                postsink->filter( newFilter );
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
            float l = project->worker.source()->length();
            
            Tools::RenderView& r = project->tools().render_view;
            r._qx -= current[0] - last[0];
            r._qz -= current[1] - last[1];
            
            if (r._qx<0) r._qx=0;
            if (r._qz<0) r._qz=0;
            if (r._qz>1) r._qz=1;
            if (r._qx>l) r._qx=l;
        }
    }

    if (infoToolButton.isDown())
    {
        GLvector current;
        if( infoToolButton.worldPos(x, y, current[0], current[1], xscale) )
        {
            Tfr::Cwt& c = Tfr::Cwt::Singleton();
            unsigned FS = project->worker.source()->sample_rate();
            float t = ((unsigned)(current[0]*FS+.5f))/(float)FS;
            //current[1] = ((unsigned)(current[1]*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            current[1] = ((current[1]*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            float f = c.compute_frequency( current[1], FS );
            float std_t = c.morlet_std_t(current[1], FS);
            float std_f = c.morlet_std_f(current[1], FS);

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
    
    project->worker.requested_fps(30);
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
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.01f,1000.0f);
    
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
            Signal::Intervals blur = _invalidRange;
            unsigned fuzzy = Tfr::Cwt::Singleton().wavelet_std_samples(project->worker.source()->sample_rate());
            blur += fuzzy;
            _invalidRange |= blur;

            blur = _invalidRange;
            blur -= fuzzy;
            _invalidRange |= blur;

            project->tools().render_view.renderer->collection()->invalidate_samples( _invalidRange );
            _invalidRange = Signal::Intervals();
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up camera position
    bool followingRecordMarker = false;
    float length = project->worker.source()->length();
    {   double limit = std::max(0.f, length - 2*Tfr::Cwt::Singleton().wavelet_std_t());
        Tools::RenderView& r = project->tools().render_view;
        if (r._qx>=_prevLimit) {
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            r._qx = std::max(r._qx, limit);
            followingRecordMarker = true;
        }
        _prevLimit = limit;

        locatePlaybackMarker();

        setupCamera();
    }

    bool wasWorking = !project->worker.todo_list().isEmpty();
    { // Render
        project->tools().render_view.renderer->collection()->next_frame(); // Discard needed blocks before this row

        Tools::RenderView& r = project->tools().render_view;
        project->tools().render_view.renderer->camera = GLvector(r._qx, r._qy, r._qz);
        project->tools().render_view.renderer->draw( 1-orthoview ); // 0.6 ms
        project->tools().render_view.renderer->drawAxes( length ); // 4.7 ms
        project->tools().selection_view.drawSelection(); // 0.1 ms

        if (wasWorking)
            drawWorking();
    }

    {   // Find things to work on (ie playback and file output)

        //    if (p && p->isUnderfed() && p->invalid_samples_left()) {
        Signal::Intervals missing_in_selection =
                project->tools().selection_model.postsinkCallback->sink()->fetch_invalid_samples();
        if (missing_in_selection)
        {
            project->worker.center = 0;
            project->worker.todo_list( missing_in_selection );

            // Request at least 1 fps. Otherwise there is a risk that CUDA
            // will screw up playback by blocking the OS and causing audio
            // starvation.
            project->worker.requested_fps(1);

            //project->worker.todo_list().print("Displaywidget - PostSink");
        } else {
            Tools::RenderView& r = project->tools().render_view;
            project->worker.center = r._qx;
            project->worker.todo_list(
                    project->tools().render_model.collectionCallback->sink()->fetch_invalid_samples());
            //project->worker.todo_list().print("Displaywidget - Collection");

            if (followingRecordMarker)
                project->worker.requested_fps(1);
        }
        Signal::Operation* first_source = project->worker.source()->root();
        Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );
        if(r != 0 && !(r->isStopped()))
        {
        	wasWorking = true;
        }
    }

    {   // Work
        bool isWorking = !project->worker.todo_list().isEmpty();

        if (wasWorking || isWorking) {
            // project->worker can be run in one or more separate threads, but if it isn't
            // execute the computations for one chunk
            if (!project->worker.isRunning()) {
                project->worker.workOne();
                QTimer::singleShot(0, this, SLOT(update())); // this will leave room for others to paint as well, calling 'update' wouldn't
            } else {
                //project->worker.todo_list().print("Work to do");
                // Wait a bit while the other thread work
                QTimer::singleShot(200, this, SLOT(update()));

                project->worker.checkForErrors();
            }

            if (!_work_timer.get())
                _work_timer.reset( new TaskTimer("Working"));
        } else {
            static unsigned workcount = 0;
            if (_work_timer) {
                _work_timer->info("Finished %u chunks, %g s. Work session #%u", project->worker.work_chunks, project->worker.work_time, workcount);
                project->worker.work_chunks = 0;
                project->worker.work_time = 0;
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
            Heightmap::pCollection c=project->tools().render_view.renderer->collection();
            c->reset(); // note, not c.reset()
            project->tools().render_view.renderer.reset();
            project->tools().render_view.renderer.reset(new Heightmap::Renderer( c ));
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
            project->tools().render_view.renderer->collection()->gc();
            tryGc++;
            //cudaThreadExit();
            cudaGetLastError();
        }
        else throw;
    }
}

void DisplayWidget::setupCamera()
{
    glLoadIdentity();
    glTranslatef( _px, _py, _pz );

    glRotatef( _rx, 1, 0, 0 );
    glRotatef( fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( _rz, 0, 0, 1 );

    glScalef(-xscale, 1, 5);

    Tools::RenderView& r = project->tools().render_view;
    glTranslatef( -r._qx, -r._qy, -r._qz );

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
    Tools::RenderView& r = project->tools().render_view;
    glVertex3f( r._qx, r._qy, r._qz );
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


void DisplayWidget::drawWaveform(Signal::pOperation /*waveform*/)
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
    cudaExtent n = chunk->waveform_data()->getNumberOfElements();
    const float* data = chunk->waveform_data()->getCpuMemory();
    
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
    TaskTimer tt("Current selection: %d", index);

    Tfr::Filter* filter = getCwtFilterHead();
    while(filter && index--)
    {
        filter = dynamic_cast<Tfr::Filter*>(filter->source().get());
    }
    if (!filter)
        return;

    MyVector* selection = project->tools().selection_model.selection;
    Filters::EllipsFilter *e = dynamic_cast<Filters::EllipsFilter*>(filter);
    if (e)
    {
        selection[0].x = e->_t1;
        selection[0].z = e->_f1;
        selection[1].x = e->_t2;
        selection[1].z = e->_f2;

        Filters::EllipsFilter *e2 = new Filters::EllipsFilter(*e );
        e2->_save_inside = true;
        e2->enabled( true );

        Signal::pOperation selectionfilter( e2 );
        selectionfilter->source( Signal::pOperation() );

        project->tools().selection_model.getPostSink()->filter( selectionfilter );
    }

    if(filter->enabled() != enabled)
    {
        filter->enabled( enabled );

        project->tools().render_view.renderer->collection()->invalidate_samples( filter->affected_samples() );
    }
    
    update();
}

void DisplayWidget::removeFilter(int index){
    TaskTimer tt("Removing filter: %d", index);

    Signal::pOperation prev;
    Signal::pOperation next = project->worker.source();

    while(next && index--)
    {
        prev = next;
        next = next->source();
    }

    if (index || !next)
        return;

    project->tools().render_view.renderer->collection()->invalidate_samples( next->affected_samples() );

    if (!prev)
    {
        setWorkerSource( next->source() );
    }
    else
    {
        prev->source( next->source() );
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
    project->tools().selection_view._playbackMarker = -1;

    if (0==project->tools().selection_model.postsinkCallback) {
        return;
    }

    // Draw playback marker
    // Find Adapters::Playback* instance
    Adapters::Playback* pb = 0;

    // todo grab it from tool
    BOOST_FOREACH( Signal::pOperation s, project->worker.callbacks() )
    {
        if ( 0 != (pb = dynamic_cast<Adapters::Playback*>( s.get() )))
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

    project->tools().selection_view._playbackMarker = pb->time();
    if (_follow_play_marker)
    {
        Tools::RenderView& r = project->tools().render_view;
        r._qx = project->tools().selection_view._playbackMarker;
    }
}

} // namespace Ui
