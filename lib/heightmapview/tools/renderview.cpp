// gl
#include "gl.h"

#include "renderview.h"

// TODO cleanup
//#include "ui/mainwindow.h"

// Sonic AWE
//#include "adapters/recorder.h"
#include "heightmap/block.h"
#include "heightmap/render/renderaxes.h"
#include "heightmap/collection.h"
#include "heightmap/uncaughtexception.h"
//#include "sawe/application.h"
//#include "sawe/project.h"
//#include "sawe/configuration.h"
//#include "ui_mainwindow.h"
//#include "support/drawwatermark.h"
#include "support/drawworking.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
//#include "toolfactory.h"
//#include "tools/recordmodel.h"
//#include "tools/support/heightmapprocessingpublisher.h"
//#include "tools/applicationerrorlogcontroller.h"
#include "tools/support/chaininfo.h"
#include "signal/processing/workers.h"
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
#include "log.h"

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
//            viewstate(new Tools::Commands::ViewState(model->project()->commandInvoker())),
            model(model),
            glwidget(0),
            rect_y_(0)
{
    // Validate rotation and set orthoview accordingly
    if (model->camera.r[0]<0) model->camera.r[0]=0;
    if (model->camera.r[0]>=90) { model->camera.r[0]=90; model->camera.orthoview.reset(1); } else model->camera.orthoview.reset(0);

    connect( this, SIGNAL(finishedWorkSection()), SLOT(finishedWorkSectionSlot()), Qt::QueuedConnection );
//    connect( viewstate.data (), SIGNAL(viewChanged(const ViewCommand*)), SLOT(redraw()));
}


RenderView::
        ~RenderView()
{
    TaskTimer tt("%s", __FUNCTION__);

    glwidget->makeCurrent();

    emit destroying();

    _render_timer.reset();
    _renderview_fbo.reset();

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
    if (Sawe::Application::global_ptr()->has_other_projects_than(this->model->project()))
        return;

    TaskInfo("cudaThreadExit()");

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

    tvector<4,float> a = model->render_settings.clear_color;
    glClearColor(a[0], a[1], a[2], a[3]);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glFrontFace( model->render_settings.left_handed_axes ? GL_CCW : GL_CW );
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
        resizeGL( QRect rect, int device_height )
{
    TIME_PAINTGL_DETAILS Log("RenderView resizeGL (x=%d y=%d w=%d h=%d) %d") % rect.left () % rect.top () % rect.width () % rect.height () % device_height;
    EXCEPTION_ASSERT_LESS(0 , rect.height ());

    glViewport( rect.x(), device_height - rect.y() - rect.height(), rect.width(), rect.height() );
    glGetIntegerv( GL_VIEWPORT,const_cast<int*>(gl_projection.viewport_matrix()) );
    rect_y_ = rect.y();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,rect.width ()/(float)rect.height (),0.01f,1000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


QRect RenderView::
        rect()
{
    const int* viewport = gl_projection.viewport_matrix();

    return QRect(viewport[0],rect_y_,viewport[2],viewport[3]);
}


void RenderView::
        paintGL()
{
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit prePaint");
        emit prePaint();
    }

    model->render_block->init();
    if (!model->render_block->isInitialized())
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

//    if (0 == "stop after 31 seconds")
//    {
//        float length = model->project()->length();
//        static unsigned frame_counter = 0;
//        TaskInfo("frame_counter = %u", ++frame_counter);
//        if (length > 30) for (static bool once=true; once; once=false)
//            QTimer::singleShot(1000, model->project()->mainWindow(), SLOT(close()));
//    }

//    Tools::RecordModel* r = model->project ()->tools ().record_model ();
//    if(r && r->recording && !r->recording.write ()->isStopped ())
//    {
//        isRecording = true;
//    }

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

    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit updatedCamera");
        emit updatedCamera();
    }

    bool onlyComputeBlocksForRenderView = false;
    Signal::OperationDesc::Extent x = model->recompute_extent ();
    { // Render
		TIME_PAINTGL_DETAILS TaskTimer tt("Render");
        float length = x.interval.get ().count() / x.sample_rate.get ();

        if (onlyComputeBlocksForRenderView)
        foreach( const Heightmap::Collection::ptr& collection, collections )
        {
            collection.write ()->next_frame(); // Discard needed blocks before this row
        }

        Support::DrawCollections(model).drawCollections( gl_projection, _renderview_fbo.get(), model->camera.r[0]>=45 ? 1 - model->camera.orthoview : 1 );

        float last_ysize = model->render_settings.last_ysize;
        glScalef(1, last_ysize*1.5 < 1. ? last_ysize*1.5 : 1. , 1); // global effect on all tools

        {
            TIME_PAINTGL_DRAW TaskTimer tt("Draw axes (%g)", length);

            bool draw_piano = model->render_settings.draw_piano;
            bool draw_hz = model->render_settings.draw_hz;
            bool draw_t = model->render_settings.draw_t;

            // apply rotation again, and make drawAxes use it
            setRotationForAxes(true);

            Heightmap::FreqAxis display_scale = model->tfr_mapping ().read()->display_scale();
            Heightmap::Render::RenderAxes(
                        model->render_settings,
                        &drawAxes_rotation,
                        display_scale
                        ).drawAxes( length );

            model->render_settings.draw_piano = draw_piano;
            model->render_settings.draw_hz = draw_hz;
            model->render_settings.draw_t = draw_t;
        }
    }

    Support::ChainInfo ci(model->chain());
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
        Heightmap::UncaughtException::handle_exception(boost::current_exception ());
    }


    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit painting");
        emit painting();
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
    for ( const auto& collection : model->collections() )
        collection->clear();

    if (model->render_block)
    {
        // model->renderer might be 0 if we're about to close the application
        // and don't bother recreating renderer if initialization has previously failed

        model->render_block->clearCaches();

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
    if (model->camera.orthoview != 1 && model->camera.orthoview != 0)
        redraw();

    glLoadIdentity();
    glTranslatef( model->camera.p[0], model->camera.p[1], model->camera.p[2] );

    glRotated( model->camera.r[0], 1, 0, 0 );
    glRotated( model->camera.effective_ry(), 0, 1, 0 );
    glRotated( model->camera.r[2], 0, 0, 1 );

    if (model->render_settings.left_handed_axes)
        glScaled(-1, 1, 1);
    else
        glRotated(-90,0,1,0);

    glScaled(model->camera.xscale, 1, model->camera.zscale);

    float a = model->camera.effective_ry();
    float dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    float dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    float dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    float dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    float limit = 5, middle=45;
    if (model->camera.r[0] < limit)
    {
        float f = 1 - model->camera.r[0] / limit;
        if (dyx<middle || dyx2<middle)
            glScalef(1,1,1-0.99999*f);
        if (dyz<middle || dyz2<middle)
            glScalef(1-0.99999*f,1,1);
    }

    glTranslated( -model->camera.q[0], -model->camera.q[1], -model->camera.q[2] );

    model->camera.orthoview.TimeStep(.08);
}


void RenderView::
        setRotationForAxes(bool setAxisVisibility)
{
    float a = model->camera.effective_ry();
    float dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    float dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    float dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    float dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    float limit = 5, middle=45;
    model->render_settings.draw_axis_at0 = 0;
    if (model->camera.r[0] < limit)
    {
        float f = 1 - model->camera.r[0] / limit;
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
                model->render_settings.draw_hz = false;
                model->render_settings.draw_piano = false;
                model->render_settings.draw_axis_at0 = dyx<middle?1:-1;
            }
            if (dyz<middle || dyz2<middle)
            {
                model->render_settings.draw_t = false;
                model->render_settings.draw_axis_at0 = dyz2<middle?1:-1;
            }
        }
    }
}


} // namespace Tools
