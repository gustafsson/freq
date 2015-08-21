// gl
#include "glstate.h"

#include "renderview.h"

// TODO cleanup
//#include "ui/mainwindow.h"

// Sonic AWE
//#include "adapters/recorder.h"
#include "heightmap/block.h"
#include "heightmap/render/renderaxes.h"
#include "heightmap/collection.h"
#include "heightmap/uncaughtexception.h"
#include "support/drawworking.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
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
#include "tvectorstring.h"

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
            drawCollections(model)
{
    // Validate rotation and set orthoview accordingly
    auto c = model->camera.write ();
    if (c->r[0]<0) c->r[0]=0;
    if (c->r[0]>=90) { c->r[0]=90; c->orthoview.reset(1); } else c->orthoview.reset(0);

    connect( this, SIGNAL(finishedWorkSection()), SLOT(finishedWorkSectionSlot()), Qt::QueuedConnection );
//    connect( viewstate.data (), SIGNAL(viewChanged(const ViewCommand*)), SLOT(redraw()));
}


RenderView::
        ~RenderView()
{
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
    GlException_CHECK_ERROR();

    const auto& v = model->gl_projection->viewport;
    glViewport (v[0], v[1], v[2], v[3]);

#ifdef GL_ES_VERSION_2_0
    GlException_SAFE_CALL( glClearDepthf(1.0f) );
#else
    GlException_SAFE_CALL( glClearDepth(1.0) );
#endif

#ifdef LEGACY_OPENGL
    GlException_SAFE_CALL( GlState::glEnable (GL_TEXTURE_2D) );
#endif

    GlException_SAFE_CALL( glDepthMask(true) );

    GlException_SAFE_CALL( GlState::glEnable (GL_DEPTH_TEST) );
    GlException_SAFE_CALL( glDepthFunc(GL_LEQUAL) );
    GlException_SAFE_CALL( glFrontFace( model->render_settings.left_handed_axes ? GL_CCW : GL_CW ) );
    GlException_SAFE_CALL( glCullFace( GL_BACK ) );
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
#ifdef LEGACY_OPENGL
    glShadeModel(GL_SMOOTH);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
#endif

#ifdef LEGACY_OPENGL
    // Antialiasing
    // This is not a recommended method for anti-aliasing. Use Multisampling instead.
    // https://www.opengl.org/wiki/Common_Mistakes#glEnable.28GL_POLYGON_SMOOTH.29
    //GlState::glEnable (GL_LINE_SMOOTH);
    //glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    //GlState::glEnable (GL_POLYGON_SMOOTH);
    //glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST);
    //GlState::glDisable (GL_POLYGON_SMOOTH);
#endif

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    GlState::glEnable (GL_BLEND);

    GlException_CHECK_ERROR();
}


void RenderView::
        defaultStates()
{
    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    GlState::glDisable (GL_DEPTH_TEST);
#ifdef LEGACY_OPENGL
    GlState::glDisable (GL_LIGHTING);
    GlState::glDisable (GL_COLOR_MATERIAL);
    GlState::glDisable (GL_LIGHT0);
    GlState::glDisable (GL_NORMALIZE);

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
    {
        tvector<4,int> viewport = model->gl_projection.read ()->viewport;
        GlException_SAFE_CALL( _renderview_fbo.reset( new GlFrameBuffer(viewport[2],viewport[3]) ) );
    }

    if (!_renderaxes)
        GlException_SAFE_CALL( _renderaxes.reset (new Heightmap::Render::RenderAxes ) );
}


void RenderView::
        resizeGL( const QRect& rect, const QSize& device )
{
    auto gl_projection = model->gl_projection.write ();

    tvector<4,int> vp(rect.x(), device.height () - rect.y() - rect.height(), rect.width(), rect.height());
    bool sameshape = vp == gl_projection->viewport && model->render_settings.device_pixel_height == device.height ();
    if (sameshape)
        return;

    TIME_PAINTGL_DETAILS Log("RenderView resizeGL (x=%d y=%d w=%d h=%d) %d") % rect.left () % rect.top () % rect.width () % rect.height () % device.height ();
    EXCEPTION_ASSERT_LESS (0, rect.height ());
    EXCEPTION_ASSERT_LESS (0, rect.width ());

    gl_projection->viewport = vp;
    model->render_settings.device_pixel_height = device.height ();

    gl_projection->modelview = matrixd::identity ();
//    glhPerspective (gl_projection->projection.v (), 45.0, rect.width ()/(double)rect.height (), 0.01, 1000.0);
    glhPerspectiveFovX (gl_projection->projection.v (), 45.0, rect.width ()/(double)rect.height (), 0.01, 1000.0);
}


QRect RenderView::
        rect()
{
    const int* viewport = model->gl_projection->viewport.v;

    int device_height = model->render_settings.device_pixel_height;
    int y_offset = device_height - viewport[1] - viewport[3];
    return QRect(viewport[0],y_offset,viewport[2],viewport[3]);
}


void RenderView::
        paintGL()
{
    if (!model->chain ())
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
    for ( auto c : collections )
    {
        // Release blocks that weren't used since last next_frame
        // Update blocks with textures from updateconsumer
        c->frame_begin();
    }

    bool update_queue_has_work = !model->update_queue()->empty ();

    setupCamera();
    glProjection gl_projection = *model->gl_projection.read ();

    {
        TIME_PAINTGL_DETAILS TaskTimer tt("emit updatedCamera");
        emit updatedCamera();
    }

    { // Render
        TIME_PAINTGL_DETAILS TaskTimer tt("Render");

        const auto c = *model->camera.read ();
        GlException_SAFE_CALL( drawCollections.drawCollections( gl_projection, _renderview_fbo.get(), c.r[0]>=45 ? 1 - c.orthoview : 1 ) );

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
            GlException_SAFE_CALL( _renderaxes->drawAxes(
                               &model->render_settings,
                               &drawAxes_rotation,
                               display_scale, length ) );
            model->render_settings.last_axes_length = length;

            model->render_settings.draw_piano = draw_piano;
            model->render_settings.draw_hz = draw_hz;
            model->render_settings.draw_t = draw_t;
        }
    }

    Support::ChainInfo ci(model->chain());
    bool isWorking = ci.hasWork () || update_queue_has_work;
#ifdef LEGACY_OPENGL
    int n_workers = ci.n_workers ();
    int dead_workers = ci.dead_workers ();

    if (isWorking || dead_workers) {
        GlException_SAFE_CALL( Support::DrawWorking::drawWorking( gl_projection.viewport[2], gl_projection.viewport[3], n_workers, dead_workers ) );
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
    GlException_SAFE_CALL( Support::DrawWatermark::drawWatermark( viewport_matrix[2], viewport_matrix[3] ) );
#endif

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

    // should rather delete this instance and recreate it
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
    const auto c = *model->camera.read ();
    glProjection gl_projection = *model->gl_projection.read ();

    if (c.orthoview != 1 && c.orthoview != 0)
        redraw();

    TIME_PAINTGL_DETAILS Log("Pos: %s, rot: %s") % c.p % c.r;

    gl_projection.modelview = matrixd::identity ();
    gl_projection.modelview *= matrixd::translate ( c.p );
    gl_projection.modelview *= matrixd::rot ( c.r[0], 1, 0, 0 );
    gl_projection.modelview *= matrixd::rot ( c.effective_ry(), 0, 1, 0 );
    gl_projection.modelview *= matrixd::rot ( c.r[2], 0, 0, 1 );

    if (model->render_settings.left_handed_axes)
        gl_projection.modelview *= matrixd::scale (-1,1,1);
    else
        gl_projection.modelview *= matrixd::rot (-90,0,1,0);

    gl_projection.modelview *= matrixd::scale (c.xscale, 1, c.zscale);

    double a = c.effective_ry();
    double dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    double dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    double dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    double dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    double limit = 5, middle=45;
    if (c.r[0] < limit)
    {
        double f = 1 - c.r[0] / limit;
        if (dyx<middle || dyx2<middle)
            gl_projection.modelview *= matrixd::scale (1,1,1-0.99999*f);
        if (dyz<middle || dyz2<middle)
            gl_projection.modelview *= matrixd::scale (1-0.99999*f,1,1);
    }

    gl_projection.modelview *= matrixd::translate ( -c.q );

    *model->gl_projection.write () = gl_projection;
    if (model->camera->orthoview.TimeStep(.08))
        redraw ();
    if (model->render_settings.log_scale.TimeStep (0.05f))
        redraw ();
}


matrixd RenderView::
        setRotationForAxes()
{
    const auto c = *model->camera.read ();
    matrixd M = matrixd::identity ();
    double a = c.effective_ry();
    double dyx2 = fabsf(fabsf(fmodf(a + 180, 360)) - 180);
    double dyx = fabsf(fabsf(fmodf(a + 0, 360)) - 180);
    double dyz2 = fabsf(fabsf(fmodf(a - 90, 360)) - 180);
    double dyz = fabsf(fabsf(fmodf(a + 90, 360)) - 180);

    double limit = 5, middle=45;
    model->render_settings.draw_axis_at0 = 0;
    if (c.r[0] < limit)
    {
        double f = 1 - c.r[0] / limit;

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
