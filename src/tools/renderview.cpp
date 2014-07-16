// gl
#include "gl.h"

#include "renderview.h"

// TODO cleanup
#include "ui/mainwindow.h"

// Sonic AWE
#include "adapters/recorder.h"
#include "heightmap/block.h"
#include "heightmap/render/renderer.h"
#include "heightmap/collection.h"
#include "sawe/application.h"
#include "sawe/project.h"
#include "sawe/configuration.h"
#include "ui_mainwindow.h"
#include "support/drawwatermark.h"
#include "support/drawworking.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "toolfactory.h"
#include "tools/recordmodel.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "tools/support/chaininfo.h"
#include "tools/applicationerrorlogcontroller.h"
#include "signal/processing/workers.h"
#include "tools/support/renderviewinfo.h"
#include "tools/support/drawcollections.h"

// gpumisc
#include "computationkernel.h"
#include "GlException.h"
#include "glPushContext.h"
#include "demangle.h"
#include "glframebuffer.h"
#include "neat_math.h"
#include "gluunproject.h"
#include "gltextureread.h"

#ifdef USE_CUDA
// cuda
#include <cuda.h> // threadexit
#endif

#include "gluproject_ios.h"

// Qt
#include <QTimer>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>
#include <QGLContext>
#include <QGraphicsView>

#include <boost/foreach.hpp>

//#define TIME_PAINTGL_DRAW
#define TIME_PAINTGL_DRAW if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            viewstate(new Tools::Commands::ViewState(model->project()->commandInvoker())),
            model(model),
            glwidget(0),
            graphicsview(0),
            _last_width(0),
            _last_height(0),
            _last_x(0),
            _last_y(0)
{
    // Validate rotation and set orthoview accordingly
    if (model->_rx<0) model->_rx=0;
    if (model->_rx>=90) { model->_rx=90; model->orthoview.reset(1); } else model->orthoview.reset(0);

    connect( Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), SLOT(clearCaches()) );
    connect( this, SIGNAL(finishedWorkSection()), SLOT(finishedWorkSectionSlot()), Qt::QueuedConnection );
    connect( model->project()->commandInvoker(), SIGNAL(projectChanged(const Command*)), SLOT(redraw()));
    connect( viewstate.data (), SIGNAL(viewChanged(const ViewCommand*)), SLOT(redraw()));
}


RenderView::
        ~RenderView()
{
    TaskTimer tt("%s", __FUNCTION__);

    glwidget->makeCurrent();

    emit destroying();

    _render_timer.reset();
    _renderview_fbo.reset();

    if (Sawe::Application::global_ptr()->has_other_projects_than(this->model->project()))
        return;

    TaskInfo("cudaThreadExit()");

//    Sawe::Application::global_ptr()->clearCaches();

    // Because the Cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
    //
    // Note though that all renderview's uses the same shared OpenGL context
    // from (Application::shared_glwidget) that are still active in other
    // Sonic AWE windows. The Cuda context will be recreated as soon as one
    // is needed. Calling 'clearCaches' above ensures that all resources are
    // released though prior to invalidating the cuda context.
    //
    // Also, see Application::clearCaches() which doesn't call cudaThreadExit
    // unless there is a current context (which is the case when clearCaches is
    // called above in this method).
//    glwidget->makeCurrent();

#ifdef USE_CUDA
    EXCEPTION_ASSERT( QGLContext::currentContext() );

    // Destroy the cuda context for this thread
    CudaException_SAFE_CALL( cudaThreadExit() );
#endif
}


void RenderView::
        setStates()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glShadeModel(GL_SMOOTH);

    tvector<4,float> a = model->renderer->render_settings.clear_color;
    glClearColor(a[0], a[1], a[2], a[3]);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glFrontFace( model->renderer->render_settings.left_handed_axes ? GL_CCW : GL_CW );
    glCullFace( GL_BACK );
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST);
        glDisable(GL_POLYGON_SMOOTH);

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
    }

    GlException_CHECK_ERROR();
}


void RenderView::
        defaultStates()
{
    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);

    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, 0.0f);
    float defaultMaterialSpecular[] = {0.0f, 0.0f, 0.0f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, defaultMaterialSpecular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
}


void RenderView::
        emitTransformChanged()
{
    emit transformChanged();
}


void RenderView::
        emitAxisChanged()
{
    emit axisChanged();
}


void RenderView::
        redraw()
{
    emit redrawSignal ();
}


void RenderView::
        initializeGL()
{
    if (!_renderview_fbo)
        _renderview_fbo.reset( new GlFrameBuffer );
}


void RenderView::
        resizeGL( int width, int height, int ratio )
{
    TIME_PAINTGL_DETAILS TaskInfo("RenderView width=%d, height=%d", width, height);
    height = height?height:1;

    QRect rect = tool_selector->parentTool()->geometry();
    rect.setWidth (rect.width ()*ratio);
    rect.setHeight (rect.height ()*ratio);
    rect.setLeft (rect.left ()*ratio);
    rect.setTop (rect.top ()*ratio);

    // Might happen during the first (few) frame during startup. Before "parentTool()" knows which size it should have.
    if (width > 1 && rect.width () > width) rect.setWidth (width);
    if (height > 1 && rect.height () > height) rect.setHeight (height);

    glViewport( rect.x(), height - rect.y() - rect.height(), rect.width(), rect.height() );
    _last_x = rect.x();
    _last_y = rect.y();
    height = rect.height();
    width = rect.width();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.01f,1000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


QRect RenderView::
        rect()
{
    //int* viewport = gl_projection->;
    return QRect();
}


void RenderView::
        paintGL()
{
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit prePaint");
        emit prePaint();
    }

    model->renderer->collection = model->tfr_mapping ().read ()->collections()[0];
    model->renderer->init();
    if (!model->renderer->isInitialized())
        return;

    float elapsed_ms = -1;

    TIME_PAINTGL_DETAILS if (_render_timer)
	    elapsed_ms = _render_timer->elapsedTime()*1000.f;
    TIME_PAINTGL_DETAILS _render_timer.reset();
    TIME_PAINTGL_DETAILS _render_timer.reset(new TaskTimer("Time since last RenderView::paintGL (%g ms, %g fps)", elapsed_ms, 1000.f/elapsed_ms));

    TIME_PAINTGL_DETAILS TaskTimer tt(".............................RenderView::paintGL.............................");

    Heightmap::TfrMapping::Collections collections = model->collections ();

    TIME_PAINTGL_DETAILS
    {
        unsigned N = collections.size();
        unsigned long sumsize = 0;
        unsigned cacheCount = 0;

        sumsize = collections[0].read ()->cacheByteSize();
        cacheCount = collections[0].read ()->cacheCount();
        for (unsigned i=1; i<N; ++i)
        {
            TaskLogIfFalse( sumsize == collections[i].read ()->cacheByteSize() );
            TaskLogIfFalse( cacheCount == collections[i].read ()->cacheCount() );
        }

        TaskInfo("Drawing (%s cache for %u*%u blocks)",
            DataStorageVoid::getMemorySizeText( N*sumsize ).c_str(),
            N, cacheCount);

        if(0) foreach( const Heightmap::Collection::ptr& c, collections )
        {
            c.read ()->printCacheSize();
        }
    }

    try
    {
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("paintGL pre check errors");
        GlException_CHECK_ERROR();
    }
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("paintGL pre sync");
        ComputationSynchronize();
    }

    if (0) {
        TIME_PAINTGL_DETAILS TaskTimer tt("Validating computation context with stft");
        // Make sure our cuda context is still alive by invoking
        // a tiny kernel. This will throw an CudaException otherwise,
        // thus resulting in restarting the cuda context.

        // If we don't ensure the context is alive before starting to
        // use various GPU resources the application will crash, for
        // instance when another RenderView is closed and releases
        // the context.
        Tfr::StftDesc a;
        a.set_approximate_chunk_size(4);
        Signal::pMonoBuffer b(new Signal::MonoBuffer(0,a.chunk_size(),1));
        (Tfr::Stft(a))(b);
    }

    // TODO move to rendercontroller
    bool isRecording = false;

    if (0 == "stop after 31 seconds")
    {
        float length = model->project()->length();
        static unsigned frame_counter = 0;
        TaskInfo("frame_counter = %u", ++frame_counter);
        if (length > 30) for (static bool once=true; once; once=false)
            QTimer::singleShot(1000, model->project()->mainWindow(), SLOT(close()));
    }

    Tools::RecordModel* r = model->project ()->tools ().record_model ();
    if(r && r->recording && !r->recording.write ()->isStopped ())
    {
        isRecording = true;
    }

    bool update_queue_has_work = !model->block_update_queue->empty ();

    if (update_queue_has_work)
        redraw (); // won't redraw right away, but enqueue an update


    // Set up camera position
    glProjection drawAxes_rotation;
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("Set up camera position");

        setupCamera();

        GLvector::T modelview_matrix[16], projection_matrix[16];
        int viewport_matrix[4];
        glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix);
        glGetIntegerv(GL_VIEWPORT, viewport_matrix);
        glGetFloatv(GL_MODELVIEW_MATRIX, modelview_matrix);
        gl_projection.update (modelview_matrix, projection_matrix, viewport_matrix);

        // drawAxes uses its own rotation
        glPushMatrixContext ctx(GL_MODELVIEW);
        setRotationForAxes(false);
        glGetFloatv(GL_MODELVIEW_MATRIX, modelview_matrix);
        drawAxes_rotation.update (modelview_matrix, projection_matrix, viewport_matrix);
    }

    // Use camera
    {
        Tools::Support::RenderViewInfo r(this);
        Heightmap::Position cursorPos = r.getPlanePos( glwidget->mapFromGlobal(QCursor::pos()) );
        model->renderer->render_settings.cursor = GLvector(cursorPos.time, 0, cursorPos.scale);
    }

    bool onlyComputeBlocksForRenderView = false;
    Signal::OperationDesc::Extent x;
    { // Render
		TIME_PAINTGL_DETAILS TaskTimer tt("Render");
        float length=0.f;

        if (onlyComputeBlocksForRenderView)
        foreach( const Heightmap::Collection::ptr& collection, collections )
        {
            collection.write ()->next_frame(); // Discard needed blocks before this row
        }

        Signal::Processing::Step::ptr step_with_new_extent;
        {
            x = model->project()->extent ();
            length = x.interval.get ().count() / x.sample_rate.get ();

            auto w = model->tfr_mapping ().write ();
            w->length( length );
            w->channels( x.number_of_channels.get () );
            w->targetSampleRate( x.sample_rate.get () );

            if (w->channels() != x.number_of_channels ||
                w->targetSampleRate() != x.sample_rate)
            {
                w->targetSampleRate( x.sample_rate.get ());
                w->channels( x.number_of_channels.get ());

                step_with_new_extent = model->target_marker ()->step().lock();
            }
        }
        if (step_with_new_extent)
            step_with_new_extent.write ()->deprecateCache(Signal::Interval::Interval_ALL);

        model->renderer->gl_projection = gl_projection;
        Support::DrawCollections(model).drawCollections( _renderview_fbo.get(), model->_rx>=45 ? 1 - model->orthoview : 1 );

        float last_ysize = model->renderer->render_settings.last_ysize;
        glScalef(1, last_ysize*1.5 < 1. ? last_ysize*1.5 : 1. , 1); // global effect on all tools

		{
			TIME_PAINTGL_DETAILS TaskTimer tt("emit painting");
			emit painting();
		}

        {
            TIME_PAINTGL_DRAW TaskTimer tt("Draw axes (%g)", length);

            bool draw_piano = model->renderer->render_settings.draw_piano;
            bool draw_hz = model->renderer->render_settings.draw_hz;
            bool draw_t = model->renderer->render_settings.draw_t;

            // apply rotation again, and make drawAxes use it
            setRotationForAxes(true);
            model->renderer->gl_projection = drawAxes_rotation;

            model->renderer->drawAxes( length ); // 4.7 ms

            model->renderer->render_settings.draw_piano = draw_piano;
            model->renderer->render_settings.draw_hz = draw_hz;
            model->renderer->render_settings.draw_t = draw_t;
        }
    }


    Support::ChainInfo ci(model->project ()->processing_chain ());
    bool isWorking = ci.hasWork () || update_queue_has_work;
    int n_workers = ci.n_workers ();
    int dead_workers = ci.dead_workers ();

    if (isWorking || isRecording || dead_workers) {
        Support::DrawWorking::drawWorking( gl_projection.viewport_matrix ()[2], gl_projection.viewport_matrix ()[3], n_workers, dead_workers );
    }

    {
        static bool hadwork = false;
        if (isWorking)
            hadwork = true;
        if (!isWorking && hadwork) {
            // Useful when debugging to close application or do something else after finishing first work chunk
            emit finishedWorkSection();
        }
    }

#if defined(TARGET_reader)
    Support::DrawWatermark::drawWatermark( viewport_matrix[2], viewport_matrix[3] );
#endif


    if (!onlyComputeBlocksForRenderView)
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("collection->next_frame");
        foreach( const Heightmap::Collection::ptr& collection, collections )
        {
            // Start looking for which blocks that are requested for the next frame.
            collection.write ()->next_frame();
        }
    }


    {
        TIME_PAINTGL_DETAILS TaskTimer tt("paintGL post check errors");
        GlException_CHECK_ERROR();
    }
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("paintGL post sync");
        ComputationCheckError();
    }

    } catch (...) {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception ());
    }


    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit postPaint");
        emit postPaint();
    }
}


void RenderView::
        clearCaches()
{
    TaskTimer tt("RenderView::clearCaches(), %p", this);
    foreach( const Heightmap::Collection::ptr& collection, model->collections() )
    {
        collection.write ()->clear();
    }

    if (model->renderer && model->renderer->isInitialized())
    {
        // model->renderer might be 0 if we're about to close the application
        // and don't bother recreating renderer if initialization has previously failed

        model->renderer->clearCaches();

        redraw();
    }
}


void RenderView::
        finishedWorkSectionSlot()
{
    //QTimer::singleShot(1000, model->project()->mainWindow(), SLOT(close()));

    //QMainWindow* mw = model->project()->mainWindow();
    //mw->setWindowState( Qt::WindowMaximized ); // WindowFullScreen
}


void RenderView::
        setupCamera()
{
    if (model->orthoview != 1 && model->orthoview != 0)
        redraw();

    glLoadIdentity();
    glTranslated( model->_px, model->_py, model->_pz );

    glRotated( model->_rx, 1, 0, 0 );
    glRotated( model->effective_ry(), 0, 1, 0 );
    glRotated( model->_rz, 0, 0, 1 );

    if (model->renderer->render_settings.left_handed_axes)
        glScaled(-1, 1, 1);
    else
        glRotated(-90,0,1,0);

    glScaled(model->xscale, 1, model->zscale);

    float a = model->effective_ry();
    float dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    float dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    float dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    float dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    float limit = 5, middle=45;
    if (model->_rx<limit)
    {
        float f = 1 - model->_rx/limit;
        if (dyx<middle || dyx2<middle)
            glScalef(1,1,1-0.99999*f);
        if (dyz<middle || dyz2<middle)
            glScalef(1-0.99999*f,1,1);
    }

    glTranslated( -model->_qx, -model->_qy, -model->_qz );

    model->orthoview.TimeStep(.08);
}


void RenderView::
        setRotationForAxes(bool setAxisVisibility)
{
    float a = model->effective_ry();
    float dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    float dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    float dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    float dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    float limit = 5, middle=45;
    model->renderer->render_settings.draw_axis_at0 = 0;
    if (model->_rx<limit)
    {
        float f = 1 - model->_rx/limit;
        if (dyx<middle)
            glRotatef(f*-90,1-dyx/middle,0,0);
        if (dyx2<middle)
            glRotatef(f*90,1-dyx2/middle,0,0);

        if (dyz<middle)
            glRotatef(f*-90,0,0,1-dyz/middle);
        if (dyz2<middle)
            glRotatef(f*90,0,0,1-dyz2/middle);

        if (setAxisVisibility)
        {
            if (dyx<middle || dyx2<middle)
            {
                model->renderer->render_settings.draw_hz = false;
                model->renderer->render_settings.draw_piano = false;
                model->renderer->render_settings.draw_axis_at0 = dyx<middle?1:-1;
            }
            if (dyz<middle || dyz2<middle)
            {
                model->renderer->render_settings.draw_t = false;
                model->renderer->render_settings.draw_axis_at0 = dyz2<middle?1:-1;
            }
        }
    }
}


} // namespace Tools
