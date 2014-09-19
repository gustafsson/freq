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
#include "gluperspective.h"

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
            rect_y_(0),
            drawCollections(model)
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
    tvector<4,float> a = model->render_settings.clear_color;
    glClearColor(a[0], a[1], a[2], a[3]);
#ifdef GL_ES_VERSION_2_0
    glClearDepthf(1.0f);
#else
    glClearDepth(1.0);
    glEnable(GL_TEXTURE_2D);
#endif
    glDepthMask(true);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glFrontFace( model->render_settings.left_handed_axes ? GL_CCW : GL_CW );
    glCullFace( GL_BACK );
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
#ifndef GL_ES_VERSION_2_0
    glShadeModel(GL_SMOOTH);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

    // Antialiasing
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST);
    glDisable(GL_POLYGON_SMOOTH);
#endif

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable(GL_BLEND);

    GlException_CHECK_ERROR();
}


void RenderView::
        defaultStates()
{
    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glDisable(GL_DEPTH_TEST);
#ifndef GL_ES_VERSION_2_0
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHT0);
    glDisable(GL_NORMALIZE);

    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, 0.0f);
    float defaultMaterialSpecular[] = {0.0f, 0.0f, 0.0f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, defaultMaterialSpecular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
#endif
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
    glProjection gl_projection = *model->gl_projection;

    tvector<4,int> vp(rect.x(), device_height - rect.y() - rect.height(), rect.width(), rect.height());
    if (vp == gl_projection.viewport && rect.y() == rect_y_)
        return;

    TIME_PAINTGL_DETAILS Log("RenderView resizeGL (x=%d y=%d w=%d h=%d) %d") % rect.left () % rect.top () % rect.width () % rect.height () % device_height;
    EXCEPTION_ASSERT_LESS(0 , rect.height ());

    gl_projection.viewport = vp;
    glViewport( vp[0], vp[1], vp[2], vp[3] );
    rect_y_ = rect.y();

    gl_projection.modelview = matrixd::identity ();
    glhPerspective (gl_projection.projection.v (), 45.0, rect.width ()/(double)rect.height (), 0.01, 1000.0);

    model->gl_projection.reset (new glProjection(gl_projection));
}


QRect RenderView::
        rect()
{
    const int* viewport = model->gl_projection->viewport.v;

    return QRect(viewport[0],rect_y_,viewport[2],viewport[3]);
}


void RenderView::
        paintGL()
{
    if (!model->chain ())
        return;

    model->render_block->init();
    if (!model->render_block->isInitialized())
        return;

    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit prePaint");
        emit prePaint();
    }

    double elapsed_ms = -1;

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
    // TODO move to rendercontroller
//    bool isRecording = false;
//    Tools::RecordModel* r = model->project ()->tools ().record_model ();
//    if(r && r->recording && !r->recording.write ()->isStopped ())
//    {
//        isRecording = true;
//    }

    bool update_queue_has_work = !model->block_update_queue->empty ();

    if (update_queue_has_work)
        redraw (); // won't redraw right away, but enqueue an update

    setupCamera();
    auto gl_projection = *model->gl_projection;

    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit updatedCamera");
        emit updatedCamera();
    }

#ifdef GL_ES_VERSION_2_0
    bool onlyComputeBlocksForRenderView = true;
#else
    bool onlyComputeBlocksForRenderView = false;
#endif
    Signal::OperationDesc::Extent x = model->recompute_extent ();
    { // Render
        TIME_PAINTGL_DETAILS TaskTimer tt("Render");

        if (onlyComputeBlocksForRenderView)
        {
            for ( auto c : collections )
                c->next_frame(); // Discard needed blocks before this row
        }

        drawCollections.drawCollections( gl_projection, _renderview_fbo.get(), model->camera.r[0]>=45 ? 1 - model->camera.orthoview : 1 );

        double last_ysize = model->render_settings.last_ysize;
        gl_projection.modelview *= matrixd::scale (1, last_ysize*1.5 < 1. ? last_ysize*1.5 : 1. , 1); // global effect on all tools

        {
            double length = model->tfr_mapping ()->length();
            TIME_PAINTGL_DETAILS TaskTimer tt("Draw axes (%g)", length);

            // setRotationForAxes messes with render_settings, this should be covered by RenderAxes
            bool draw_piano = model->render_settings.draw_piano;
            bool draw_hz = model->render_settings.draw_hz;
            bool draw_t = model->render_settings.draw_t;

            // apply rotation again, and make drawAxes use it
            glProjection drawAxes_rotation = gl_projection;
            drawAxes_rotation.modelview *= setRotationForAxes();

            Heightmap::FreqAxis display_scale = model->tfr_mapping ().read()->display_scale();
            Heightmap::Render::RenderAxes(
                    model->render_settings,
                    &drawAxes_rotation,
                    display_scale
                    ).drawAxes( length );
            model->render_settings.last_axes_length = length;

            model->render_settings.draw_piano = draw_piano;
            model->render_settings.draw_hz = draw_hz;
            model->render_settings.draw_t = draw_t;
        }
    }

    Support::ChainInfo ci(model->chain());
    bool isWorking = ci.hasWork () || update_queue_has_work;
#ifndef GL_ES_VERSION_2_0
    int n_workers = ci.n_workers ();
    int dead_workers = ci.dead_workers ();

    if (isWorking || dead_workers) {
        Support::DrawWorking::drawWorking( gl_projection.viewport[2], gl_projection.viewport[3], n_workers, dead_workers );
    }
#endif

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

        for ( auto c : collections )
            // Start looking for which blocks that are requested for the next frame.
            c->next_frame();
    }

    } catch (...) {
        Heightmap::UncaughtException::handle_exception(boost::current_exception ());
    }


    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit painting");
        emit painting();
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
    glProjection gl_projection = *model->gl_projection;

    if (model->camera.orthoview != 1 && model->camera.orthoview != 0)
        redraw();

    gl_projection.modelview = matrixd::identity ();
    gl_projection.modelview *= matrixd::translate ( model->camera.p );
    gl_projection.modelview *= matrixd::rot ( model->camera.r[0], 1, 0, 0 );
    gl_projection.modelview *= matrixd::rot ( model->camera.effective_ry(), 0, 1, 0 );
    gl_projection.modelview *= matrixd::rot ( model->camera.r[2], 0, 0, 1 );

    if (model->render_settings.left_handed_axes)
        gl_projection.modelview *= matrixd::scale (-1,1,1);
    else
        gl_projection.modelview *= matrixd::rot (-90,0,1,0);

    gl_projection.modelview *= matrixd::scale (model->camera.xscale, 1, model->camera.zscale);

    double a = model->camera.effective_ry();
    double dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    double dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    double dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    double dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    double limit = 5, middle=45;
    if (model->camera.r[0] < limit)
    {
        double f = 1 - model->camera.r[0] / limit;
        if (dyx<middle || dyx2<middle)
            gl_projection.modelview *= matrixd::scale (1,1,1-0.99999*f);
        if (dyz<middle || dyz2<middle)
            gl_projection.modelview *= matrixd::scale (1-0.99999*f,1,1);
    }

    gl_projection.modelview *= matrixd::translate ( -model->camera.q );

    model->gl_projection.reset (new glProjection(gl_projection));
    if (model->camera.orthoview.TimeStep(.08))
        redraw ();
    if (model->render_settings.log_scale.TimeStep (0.05f))
        redraw ();
}


matrixd RenderView::
        setRotationForAxes()
{
    matrixd M = matrixd::identity ();
    double a = model->camera.effective_ry();
    double dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    double dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    double dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    double dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    double limit = 5, middle=45;
    model->render_settings.draw_axis_at0 = 0;
    if (model->camera.r[0] < limit)
    {
        double f = 1 - model->camera.r[0] / limit;

        if (dyx<middle)
            M *= matrixd::rot (f*-90, 1-dyx/middle,0,0);
        if (dyx2<middle)
            M *= matrixd::rot (f*90, 1-dyx2/middle,0,0);

        if (dyz<middle)
            M *= matrixd::rot (f*-90,0,0,1-dyz/middle);
        if (dyz2<middle)
            M *= matrixd::rot (f*90,0,0,1-dyz2/middle);

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
    return M;
}


} // namespace Tools
