#include "displaywidget.h"

// QT
#include <QKeyEvent>
#include <QToolTip>

// Boost
#include <boost/foreach.hpp>

// Sonic AWE
#include "adapters/playback.h"
#include "adapters/microphonerecorder.h"
#include "signal/operation-composite.h"
#include "signal/operation-basic.h"
#include "adapters/matlabfilter.h"
#include "adapters/writewav.h"
#include "filters/filters.h"
#include "tfr/cwt.h"
#include "tools/toolfactory.h"
#include "tools/renderview.h"


namespace Ui {


using namespace std;


DisplayWidget::
        DisplayWidget(
                Sawe::Project* project,
                Tools::RenderView* render_view,
                Tools::RenderModel* render_model )
: QWidget(),
  lastKey(0),
  orthoview(1),
//  _record_update(false),
  project( project ),
  _render_model( render_model ),
  _render_view( render_view ),
  _follow_play_marker( false ),
  _prevX(0), _prevY(0), _targetQ(0),
  _selectionActive(true),
  _navigationActive(false),
  _infoToolActive(false),
  selecting(false)
{
    yscale = Yscale_LogLinear;
    //timeOut();

    Tools::RenderView &r = *_render_view;
    if (r._rx<0) r._rx=0;
    if (r._rx>90) { r._rx=90; orthoview=1; }
    if (0<orthoview && r._rx<90) { r._rx=90; orthoview=0; }
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

    _render_view->update();
}


void DisplayWidget::receiveFollowPlayMarker( bool v )
{
    _follow_play_marker = v;
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
    _render_view->update();
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
    _render_view->update();
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
        // _render_model->collection->add_expected_samples( f->affected_samples() );
    }

    postsink_filter->source( project->worker.source() );
    setWorkerSource( postsink_filter );
    _render_view->update();
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
    _render_model->collection->invalidate_samples(sid);

    // Update stream
    b->source(remove);

    setWorkerSource();
    _render_view->update();
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

        _render_model->collection->invalidate_samples(sid);
        _render_view->update();
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
        _render_model->collection->invalidate_samples(sid);

        // update stream
        b->source(moveSelection );

        setWorkerSource();
        _render_view->update();
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
    _render_view->update();
    _render_model->collection->invalidate_samples(Signal::Intervals::Intervals_ALL);
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


    _render_model->collection->invalidate_samples(b->affected_samples());

    setWorkerSource();
    _render_view->update();
}


void DisplayWidget::
        receiveTonalizeFilter(bool)
{
    Signal::pOperation tonalize( new Filters::TonalizeFilter());
    tonalize->source( project->worker.source() );

    setWorkerSource(tonalize);

    _render_model->collection->invalidate_samples( tonalize->affected_samples());

    _render_view->update();
}


void DisplayWidget::
        receiveReassignFilter(bool)
{
    Signal::pOperation reassign( new Filters::ReassignFilter());
    reassign->source(project->worker.source());
    setWorkerSource(reassign);

    _render_model->collection->invalidate_samples( reassign->affected_samples() );

    _render_view->update();
}


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
    //_render_view->update();
}*/


Tfr::CwtFilter* DisplayWidget::
        getCwtFilterHead()
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


void DisplayWidget::
        mousePressEvent ( QMouseEvent * e )
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
    
    _prevX = e->x(),
    _prevY = e->y();

    _render_view->update();
}

void DisplayWidget::
        mouseReleaseEvent ( QMouseEvent * e )
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
    _render_view->update();
}


void DisplayWidget::
        wheelEvent ( QWheelEvent *e )
{
    Tools::RenderView &r = *_render_view;
    float ps = 0.0005;
    float rs = 0.08;
    if( e->orientation() == Qt::Horizontal )
    {
        if(e->modifiers().testFlag(Qt::ShiftModifier))
            r.xscale *= (1-ps * e->delta());
        else
            r._ry -= rs * e->delta();
    }
    else
    {
		if(e->modifiers().testFlag(Qt::ShiftModifier))
            r.xscale *= (1-ps * e->delta());
        else
            r._pz *= (1+ps * e->delta());

        if (r._pz<-40) r._pz = -40;
        if (r._pz>-.1) r._pz = -.1;
        //_pz -= ps * e->delta();
        
        //_rx -= ps * e->delta();
    }
    
    _render_view->update();
}


void DisplayWidget::
        mouseMoveEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *_render_view;
    r.makeCurrent();

    float rs = 0.2;
    
    int x = e->x(), y = this->height() - e->y();
//    TaskTimer tt("moving");
    
    if ( selectionButton.isDown() )
    {
        MyVector* selection = project->tools().selection_model.selection;
        GLdouble p[2];
        if (selectionButton.worldPos(x, y, p[0], p[1], r.xscale))
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
        r._ry += (1-orthoview)*rs * rotateButton.deltaX( x );
        r._rx -= rs * rotateButton.deltaY( y );
        if (r._rx<10) r._rx=10;
        if (r._rx>90) { r._rx=90; orthoview=1; }
        if (0<orthoview && r._rx<90) { r._rx=90; orthoview=0; }
        
    }

    if( moveButton.isDown() )
    {
        //Controlling the position with the right button.
        GLvector last, current;
        if( moveButton.worldPos(last[0], last[1], r.xscale) &&
            moveButton.worldPos(x, y, current[0], current[1], r.xscale) )
        {
            float l = project->worker.source()->length();
            
            Tools::RenderView& r = *_render_view;
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
        if( infoToolButton.worldPos(x, y, current[0], current[1], r.xscale) )
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
    _render_view->update();
}


// todo remove
//static void printQGLFormat(const QGLFormat& f, std::string title)
//{
//    TaskTimer tt("QGLFormat %s", title.c_str());
//    tt.info("accum=%d",f.accum());
//    tt.info("accumBufferSize=%d",f.accumBufferSize());
//    tt.info("alpha=%d",f.alpha());
//    tt.info("alphaBufferSize=%d",f.alphaBufferSize());
//    tt.info("blueBufferSize=%d",f.blueBufferSize());
//    tt.info("depth=%d",f.depth());
//    tt.info("depthBufferSize=%d",f.depthBufferSize());
//    tt.info("directRendering=%d",f.directRendering());
//    tt.info("doubleBuffer=%d",f.doubleBuffer());
//    tt.info("greenBufferSize=%d",f.greenBufferSize());
//    tt.info("hasOverlay=%d",f.hasOverlay());
//    tt.info("redBufferSize=%d",f.redBufferSize());
//    tt.info("rgba=%d",f.rgba());
//    tt.info("sampleBuffers=%d",f.sampleBuffers());
//    tt.info("samples=%d",f.samples());
//    tt.info("stencil=%d",f.stencil());
//    tt.info("stencilBufferSize=%d",f.stencilBufferSize());
//    tt.info("stereo=%d",f.stereo());
//    tt.info("swapInterval=%d",f.swapInterval());
//    tt.info("");
//    tt.info("hasOpenGL=%d",f.hasOpenGL());
//    tt.info("hasOpenGLOverlays=%d",f.hasOpenGLOverlays());
//    QGLFormat::OpenGLVersionFlags flag = f.openGLVersionFlags();
//    tt.info("OpenGL_Version_None=%d", QGLFormat::OpenGL_Version_None == flag);
//    tt.info("OpenGL_Version_1_1=%d", QGLFormat::OpenGL_Version_1_1 & flag);
//    tt.info("OpenGL_Version_1_2=%d", QGLFormat::OpenGL_Version_1_2 & flag);
//    tt.info("OpenGL_Version_1_3=%d", QGLFormat::OpenGL_Version_1_3 & flag);
//    tt.info("OpenGL_Version_1_4=%d", QGLFormat::OpenGL_Version_1_4 & flag);
//    tt.info("OpenGL_Version_1_5=%d", QGLFormat::OpenGL_Version_1_5 & flag);
//    tt.info("OpenGL_Version_2_0=%d", QGLFormat::OpenGL_Version_2_0 & flag);
//    tt.info("OpenGL_Version_2_1=%d", QGLFormat::OpenGL_Version_2_1 & flag);
//    tt.info("OpenGL_Version_3_0=%d", QGLFormat::OpenGL_Version_3_0 & flag);
//    tt.info("OpenGL_ES_CommonLite_Version_1_0=%d", QGLFormat::OpenGL_ES_CommonLite_Version_1_0 & flag);
//    tt.info("OpenGL_ES_Common_Version_1_0=%d", QGLFormat::OpenGL_ES_Common_Version_1_0 & flag);
//    tt.info("OpenGL_ES_CommonLite_Version_1_1=%d", QGLFormat::OpenGL_ES_CommonLite_Version_1_1 & flag);
//    tt.info("OpenGL_ES_Common_Version_1_1=%d", QGLFormat::OpenGL_ES_Common_Version_1_1 & flag);
//    tt.info("OpenGL_ES_Version_2_0=%d", QGLFormat::OpenGL_ES_Version_2_0 & flag);
//}


// todo remove
//static void printQGLWidget(const QGLWidget& w, std::string title)
//{
//    TaskTimer tt("QGLWidget %s", title.c_str());
//    tt.info("doubleBuffer=%d", w.doubleBuffer());
//    tt.info("isSharing=%d", w.isSharing());
//    tt.info("isValid=%d", w.isValid());
//    printQGLFormat( w.format(), "");
//}


void DisplayWidget::
        setSelection(int index, bool enabled)
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

        _render_model->collection->invalidate_samples( filter->affected_samples() );
    }
    
    _render_view->update();
}


void DisplayWidget::
        removeFilter(int index)
{
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

    _render_model->collection->invalidate_samples( next->affected_samples() );

    if (!prev)
    {
        setWorkerSource( next->source() );
    }
    else
    {
        prev->source( next->source() );
    }

    _render_view->update();
    setWorkerSource();
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
        Tools::RenderView& r = *_render_view;
        r._qx = project->tools().selection_view._playbackMarker;
    }
}

} // namespace Ui
