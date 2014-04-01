// gl
#include "gl.h"

#include "renderview.h"

// TODO cleanup
#include "ui/mainwindow.h"

// Sonic AWE
#include "adapters/recorder.h"
#include "heightmap/renderer.h"
#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "heightmap/collection.h"
#include "heightmap/blocks/chunkmerger.h"
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

// gpumisc
#include "computationkernel.h"
#include "GlException.h"
#include "glPushContext.h"
#include "demangle.h"
#include "glframebuffer.h"
#include "neat_math.h"
#include "gluunproject.h"

#ifdef USE_CUDA
// cuda
#include <cuda.h> // threadexit
#endif

// Qt
#include <QTimer>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>
#include <QGLContext>
#include <QGraphicsView>

#include <boost/foreach.hpp>

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

//#define TIME_PAINTGL_DRAW
#define TIME_PAINTGL_DRAW if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

//#define DEBUG_EVENTS
#define DEBUG_EVENTS if(0)

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
            last_ysize(1),
            viewstate(new Tools::Commands::ViewState(model->project()->commandInvoker())),
            model(model),
            glwidget(0),
            graphicsview(0),
            _inited(false),
            _last_width(0),
            _last_height(0),
            _last_x(0),
            _last_y(0),
            _try_gc(0),
            _target_fps(10.0f),
            _last_update_size(std::numeric_limits<Signal::UnsignedIntervalType>::max())
{
    // Validate rotation and set orthoview accordingly
    if (model->_rx<0) model->_rx=0;
    if (model->_rx>=90) { model->_rx=90; model->orthoview.reset(1); } else model->orthoview.reset(0);

    connect( Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), SLOT(clearCaches()) );
    connect( this, SIGNAL(finishedWorkSection()), SLOT(finishedWorkSectionSlot()), Qt::QueuedConnection );
    connect( this, SIGNAL(sceneRectChanged ( const QRectF & )), SLOT(redraw()) );
    connect( model->project()->commandInvoker(), SIGNAL(projectChanged(const Command*)), SLOT(redraw()));
    connect( viewstate.data (), SIGNAL(viewChanged(const ViewCommand*)), SLOT(restartUpdateTimer()));

    _update_timer = new QTimer;
    _update_timer->setSingleShot( true );

    connect( this, SIGNAL(postUpdate()), SLOT(restartUpdateTimer()), Qt::QueuedConnection );
    connect( _update_timer.data(), SIGNAL(timeout()), SLOT(update()), Qt::QueuedConnection );
}


RenderView::
        ~RenderView()
{
    TaskTimer tt("%s", __FUNCTION__);

    delete _update_timer;

    glwidget->makeCurrent();

    emit destroying();

    _render_timer.reset();
    _renderview_fbo.reset();

    QGraphicsScene::clear();

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


bool RenderView::
        event ( QEvent * e )
{
    DEBUG_EVENTS TaskTimer tt("RenderView event %s %d", vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsScene::event(e);
    DEBUG_EVENTS TaskTimer("RenderView event %s info %d %d", vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}

bool RenderView::
        eventFilter(QObject* o, QEvent* e)
{
    DEBUG_EVENTS TaskTimer tt("RenderView eventFilter %s %s %d", vartype(*o).c_str(), vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsScene::eventFilter(o, e);
    DEBUG_EVENTS TaskTimer("RenderView eventFilter %s %s info %d %d", vartype(*o).c_str(), vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}

void RenderView::
        mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("RenderView mousePressEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mousePressEvent(e);
    DEBUG_EVENTS TaskTimer("RenderView mousePressEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void RenderView::
        mouseMoveEvent(QGraphicsSceneMouseEvent *e)
{
    if (model->renderer->render_settings.draw_cursor_marker)
        update();

    redraw();

    DEBUG_EVENTS TaskTimer tt("RenderView mouseMoveEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mouseMoveEvent(e);
    DEBUG_EVENTS TaskTimer("RenderView mouseMoveEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void RenderView::
        mouseReleaseEvent(QGraphicsSceneMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("RenderView mouseReleaseEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mouseReleaseEvent(e);
    DEBUG_EVENTS TaskTimer("RenderView mouseReleaseEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}


void RenderView::
        drawBackground(QPainter *painter, const QRectF &)
{
    double T = _last_frame.elapsedAndRestart();
    TIME_PAINTGL TaskTimer tt("%g ms", T*1e3);

    painter->beginNativePainting();

    glMatrixMode(GL_MODELVIEW);

    try { {
        glPushAttribContext attribs;
        glPushMatrixContext pmcp(GL_PROJECTION);
        glPushMatrixContext pmcm(GL_MODELVIEW);

        if (!_inited)
            initializeGL();

		if (painter->device())
		{
            unsigned w = painter->device()->width();
            unsigned h = painter->device()->height();
            w *= painter->device ()->devicePixelRatio();
            h *= painter->device ()->devicePixelRatio();
            if (w != _last_width || h != _last_height)
                redraw();
            _last_width = w;
            _last_height = h;
		}

        setStates();

        {
            TIME_PAINTGL_DETAILS TaskTimer tt("glClear");
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        {
            TIME_PAINTGL_DETAILS TaskTimer tt("emit prePaint");
            emit prePaint();
        }

        resizeGL(_last_width, _last_height, painter->device ()->devicePixelRatio() );

        paintGL();

        defaultStates();

        }
        GlException_CHECK_ERROR();
    } catch (const std::exception& x) {
        TaskInfo("");
        TaskInfo(boost::format("std::exception\n%s") % boost::diagnostic_information(x));
        TaskInfo("");
    } catch (...) {
        TaskInfo(boost::format("Not an std::exception\n%s") % boost::current_exception_diagnostic_information ());
    }

    painter->endNativePainting();
}


void RenderView::
        drawForeground(QPainter *painter, const QRectF &)
{
    painter->beginNativePainting();
    setStates();

    emit paintingForeground();

    defaultStates();

    painter->endNativePainting();
}


float RenderView::
        getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref, float* pick_local_max, bool fetch_interpolation, bool* is_valid_value )
{
    if (is_valid_value)
        *is_valid_value = true;

    if (pos.time < 0 || pos.scale < 0 || pos.scale >= 1 || pos.time > model->project()->length())
        return 0;

    if (is_valid_value)
        *is_valid_value = false;

    Heightmap::Reference myref;
    if (!ref)
    {
        ref = &myref;
        ref->block_index[0] = (unsigned)-1;
    }
    if (ref->block_index[0] == (unsigned)-1)
    {
        *ref = findRefAtCurrentZoomLevel( pos );
    }

    Heightmap::RegionFactory rr(model->tfr_mapping ().read ()->block_layout ());
    Heightmap::Region r = rr(*ref);

    ref->block_index[0] = pos.time / r.time();
    ref->block_index[1] = pos.scale / r.scale();
    r = rr(*ref);

    Heightmap::Collection::Ptr collection = model->collections()[0];
    Heightmap::pBlock block = collection.read ()->getBlock( *ref );
    Heightmap::ReferenceInfo ri(block->referenceInfo ());
    if (!block)
        return 0;
    if (is_valid_value)
    {
        Signal::IntervalType s = pos.time * ri.sample_rate();
        Signal::Intervals I = Signal::Intervals(s, s+1);
        I &= ri.getInterval ();
        if (I.empty())
        {
            *is_valid_value = true;
        } else
            return 0;
    }

    DataStorage<float>::Ptr blockData = block->glblock->height()->data;

    float* data = blockData->getCpuMemory();
    Heightmap::BlockLayout block_layout = model->tfr_mapping ().read ()->block_layout();
    unsigned w = block_layout.texels_per_row ();
    unsigned h = block_layout.texels_per_column ();
    unsigned x0 = (pos.time-r.a.time)/r.time()*(w-1) + .5f;
    float    yf = (pos.scale-r.a.scale)/r.scale()*(h-1);
    unsigned y0 = yf + .5f;

    EXCEPTION_ASSERT( x0 < w );
    EXCEPTION_ASSERT( y0 < h );
    float v;

    if (!pick_local_max && !fetch_interpolation)
    {
        v = data[ x0 + y0*w ];
    }
    else
    {
        unsigned yb = y0;
        if (0==yb) yb++;
        if (h==yb+1) yb--;
        float v1 = data[ x0 + (yb-1)*w ];
        float v2 = data[ x0 + yb*w ];
        float v3 = data[ x0 + (yb+1)*w ];

        // v1 = a*(-1)^2 + b*(-1) + c
        // v2 = a*(0)^2 + b*(0) + c
        // v3 = a*(1)^2 + b*(1) + c
        float k = 0.5f*v1 - v2 + 0.5f*v3;
        float p = -0.5f*v1 + 0.5f*v3;
        float q = v2;

        float m0;
        if (fetch_interpolation)
        {
            m0 = yf - yb;
        }
        else
        {
            // fetch max
            m0 = -p/(2*k);
        }

        if (m0 > -2 && m0 < 2)
        {
            v = k*m0*m0 + p*m0 + q;
            if (pick_local_max)
                *pick_local_max = r.a.scale + r.scale()*(y0 + m0)/(h-1);
        }
        else
        {
            v = v2;
            if (pick_local_max)
                *pick_local_max = r.a.scale + r.scale()*(y0)/(h-1);
        }

        float testmax;
        float testv = quad_interpol( y0, data + x0, h, w, &testmax );
        float testlocalmax = r.a.scale + r.scale()*(testmax)/(h-1);

        EXCEPTION_ASSERT( testv == v );
        if (pick_local_max)
            EXCEPTION_ASSERT( testlocalmax == *pick_local_max );
    }

    return v;
}


Heightmap::Reference RenderView::
        findRefAtCurrentZoomLevel(Heightmap::Position p)
{
    model->renderer->gl_projection.update ();
//    memcpy( model->renderer->gl_projection.viewport_matrix (), viewport_matrix, sizeof(viewport_matrix));
//    memcpy( model->renderer->gl_projection.modelview_matrix (), modelview_matrix, sizeof(modelview_matrix));
//    memcpy( model->renderer->gl_projection.projection_matrix (), projection_matrix, sizeof(projection_matrix));

    model->renderer->collection = model->collections()[0];

    return model->renderer->findRefAtCurrentZoomLevel( p );
}


QPointF RenderView::
        getScreenPos( Heightmap::Position pos, double* dist, bool use_heightmap_value )
{
    GLdouble objY = 0;
    if ((1 != model->orthoview || model->_rx!=90) && use_heightmap_value)
        objY = getHeightmapValue(pos) * model->renderer->render_settings.y_scale * last_ysize;

    GLdouble winX, winY, winZ;
    gluProject( pos.time, objY, pos.scale,
                modelview_matrix, projection_matrix, viewport_matrix,
                &winX, &winY, &winZ);

    if (dist)
    {
        GLint const* const& vp = viewport_matrix;
        float z0 = .1, z1=.2;
        GLvector projectionPlane = ::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z0), modelview_matrix, projection_matrix, vp );
        GLvector projectionNormal = (::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z1), modelview_matrix, projection_matrix, vp ) - projectionPlane);

        GLvector p;
        p[0] = pos.time;
        p[1] = 0;//objY;
        p[2] = pos.scale;

        GLvector d = p-projectionPlane;
        projectionNormal[0] *= model->xscale;
        projectionNormal[2] *= last_ysize;
        d[0] *= model->xscale;
        d[2] *= last_ysize;

        projectionNormal = projectionNormal.Normalized();

        *dist = d%projectionNormal;
    }

    int r = glwidget->devicePixelRatio ();
    return QPointF( winX, _last_height-1 - winY ) /= r;
}


QPointF RenderView::
        getWidgetPos( Heightmap::Position pos, double* dist, bool use_heightmap_value )
{
    QPointF pt = getScreenPos(pos, dist, use_heightmap_value);
    pt -= QPointF(_last_x, _last_y);
    return pt;
}


Heightmap::Position RenderView::
        getHeightmapPos( QPointF widget_pos, bool useRenderViewContext )
{
    if (1 == model->orthoview)
        return getPlanePos(widget_pos, 0, useRenderViewContext);

    TaskTimer tt("RenderView::getHeightmapPos(%g, %g) Newton raphson", widget_pos.x(), widget_pos.y());
    widget_pos *= glwidget->devicePixelRatio ();

    QPointF pos;
    pos.setX( widget_pos.x() + _last_x );
    pos.setY( _last_height - 1 - widget_pos.y() + _last_y );

    GLdouble* m = this->modelview_matrix, *proj = this->projection_matrix;
    GLint* vp = this->viewport_matrix;
    GLdouble other_m[16], other_proj[16];
    GLint other_vp[4];
    if (!useRenderViewContext)
    {
        glGetDoublev(GL_MODELVIEW_MATRIX, other_m);
        glGetDoublev(GL_PROJECTION_MATRIX, other_proj);
        glGetIntegerv(GL_VIEWPORT, other_vp);
        m = other_m;
        proj = other_proj;
        vp = other_vp;
    }

    GLdouble objX1, objY1, objZ1;
    gluUnProject( pos.x(), pos.y(), 0.1,
                m, proj, vp,
                &objX1, &objY1, &objZ1);

    GLdouble objX2, objY2, objZ2;
    gluUnProject( pos.x(), pos.y(), 0.2,
                m, proj, vp,
                &objX2, &objY2, &objZ2);

    Heightmap::Position p;
    double y = 0;

    double prevs = 1;
    // Newton raphson with naive damping
    for (int i=0; i<50; ++i)
    {
        double s = (y-objY1)/(objY2-objY1);
        double k = 1./8;
        s = (1-k)*prevs + k*s;

        Heightmap::Position q;
        q.time = objX1 + s * (objX2-objX1);
        q.scale = objZ1 + s * (objZ2-objZ1);

        Heightmap::Position d(
                (q.time-p.time)*model->xscale,
                (q.scale-p.scale)*model->zscale);

        QPointF r = getWidgetPos( q, 0 );
        d.time = r.x() - widget_pos.x();
        d.scale = r.y() - widget_pos.y();
        float e = d.time*d.time + d.scale*d.scale;
        p = q;
        if (e < 0.4) // || prevs > s)
            break;
        //if (e < 1e-5 )
        //    break;

        y = getHeightmapValue(p) * model->renderer->render_settings.y_scale * 4 * last_ysize;
        tt.info("(%g, %g) %g is Screen(%g, %g), s = %g", p.time, p.scale, y, r.x(), r.y(), s);
        prevs = s;
    }

    y = getHeightmapValue(p) * model->renderer->render_settings.y_scale * 4 * last_ysize;
    TaskInfo("Screen(%g, %g) projects at Heightmap(%g, %g, %g)", widget_pos.x(), widget_pos.y(), p.time, p.scale, y);
    QPointF r = getWidgetPos( p, 0 );
    TaskInfo("Heightmap(%g, %g) projects at Screen(%g, %g)", p.time, p.scale, r.x(), r.y() );
    return p;
}

Heightmap::Position RenderView::
        getPlanePos( QPointF pos, bool* success, bool useRenderViewContext )
{
    pos *= glwidget->devicePixelRatio ();

    pos.setX( pos.x() + _last_x );
    pos.setY( _last_height - 1 - pos.y() + _last_y );

    GLdouble* m = this->modelview_matrix, *proj = this->projection_matrix;
    GLint* vp = this->viewport_matrix;
    GLdouble other_m[16], other_proj[16];
    GLint other_vp[4];
    if (!useRenderViewContext)
    {
        glGetDoublev(GL_MODELVIEW_MATRIX, other_m);
        glGetDoublev(GL_PROJECTION_MATRIX, other_proj);
        glGetIntegerv(GL_VIEWPORT, other_vp);
        m = other_m;
        proj = other_proj;
        vp = other_vp;
    }

    GLdouble objX1, objY1, objZ1;
    gluUnProject( pos.x(), pos.y(), 0.1,
                m, proj, vp,
                &objX1, &objY1, &objZ1);

    GLdouble objX2, objY2, objZ2;
    gluUnProject( pos.x(), pos.y(), 0.2,
                m, proj, vp,
                &objX2, &objY2, &objZ2);

    double ylevel = 0;
    double s = (ylevel-objY1)/(objY2-objY1);

    if (0==objY2-objY1)
        s = 0;

    Heightmap::Position p;
    p.time = objX1 + s * (objX2-objX1);
    p.scale = objZ1 + s * (objZ2-objZ1);

    if (success) *success=true;

    float minAngle = 3;
    if(success)
    {
        if( s < 0)
            if (success) *success=false;

        float L = sqrt((objX1-objX2)*(objX1-objX2)*model->xscale*model->xscale
                       +(objY1-objY2)*(objY1-objY2)
                       +(objZ1-objZ2)*(objZ1-objZ2)*model->zscale*model->zscale);
        if (objY1-objY2 < sin(minAngle *(M_PI/180)) * L )
            if (success) *success=false;
    }

    return p;
}


QPointF RenderView::
        widget_coordinates( QPointF window_coordinates )
{
    return window_coordinates - QPointF(_last_x, _last_y);
}


QPointF RenderView::
        window_coordinates( QPointF widget_coordinates )
{
    return widget_coordinates + QPointF(_last_x, _last_y);
}


void RenderView::
        drawCollections(GlFrameBuffer* fbo, float yscale)
{
    TIME_PAINTGL_DRAW TaskTimer tt2("Drawing...");
    GlException_CHECK_ERROR();

    unsigned N = model->collections().size();
    if (N != channel_colors.size ())
        computeChannelColors ();

    TIME_PAINTGL_DETAILS ComputationCheckError();

    // Draw the first channel without a frame buffer
    model->renderer->render_settings.camera = GLvector(model->_qx, model->_qy, model->_qz);
    model->renderer->render_settings.cameraRotation = GLvector(model->_rx, model->_ry, model->_rz);

    Heightmap::Position cursorPos = getPlanePos( glwidget->mapFromGlobal(QCursor::pos()) );
    model->renderer->render_settings.cursor = GLvector(cursorPos.time, 0, cursorPos.scale);

    // When rendering to fbo, draw to the entire fbo, then update the current
    // viewport.
    GLint current_viewport[4];
    glGetIntegerv(GL_VIEWPORT, current_viewport);
    GLint viewportWidth = current_viewport[2],
          viewportHeight = current_viewport[3];


    TIME_PAINTGL_DETAILS TaskTimer tt("Viewport (%u, %u, %u, %u)",
        current_viewport[0], current_viewport[1],
        current_viewport[2], current_viewport[3]);

    unsigned i=0;

    const Heightmap::TfrMapping::Collections collections = model->collections ();

    // draw the first without fbo
    for (; i < N; ++i)
    {
        if (!collections[i].read ()->isVisible())
            continue;

        drawCollection(i, yscale);
        ++i;
        break;
    }


    bool hasValidatedFboSize = false;

    for (; i<N; ++i)
    {
        if (!collections[i].read ()->isVisible())
            continue;

        if (!hasValidatedFboSize)
        {
            // drawCollections is called for several different viewports each frame.
            // GlFrameBuffer will query the current viewport to determine the size
            // of the fbo for this iteration.
            if (viewportWidth > fbo->getWidth ()
                || viewportHeight > fbo->getHeight()
                || viewportWidth*2 < fbo->getWidth()
                || viewportHeight*2 < fbo->getHeight())
            {
                fbo->recreate();
            }

            hasValidatedFboSize = true;
        }

        GlException_CHECK_ERROR();

        {
            GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding();
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
            glViewport(0, 0, viewportWidth, viewportHeight);

            drawCollection(i, yscale);
        }

        glViewport(current_viewport[0], current_viewport[1],
                   current_viewport[2], current_viewport[3]);

        glPushMatrixContext mpc( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(0,1,0,1,-10,10);
        glPushMatrixContext mc( GL_MODELVIEW );
        glLoadIdentity();

        glBlendFunc( GL_DST_COLOR, GL_ZERO );

        glDisable(GL_DEPTH_TEST);

        glColor4f(1,1,1,1);
        GlTexture t(fbo->getGlTexture());
        GlTexture::ScopeBinding texObjBinding = t.getScopeBinding();

        glBegin(GL_TRIANGLE_STRIP);
            float tx = viewportWidth/(float)fbo->getWidth();
            float ty = viewportHeight/(float)fbo->getHeight();
            glTexCoord2f(0,0); glVertex2f(0,0);
            glTexCoord2f(tx,0); glVertex2f(1,0);
            glTexCoord2f(0,ty); glVertex2f(0,1);
            glTexCoord2f(tx,ty); glVertex2f(1,1);
        glEnd();

        glEnable(GL_DEPTH_TEST);

        GlException_CHECK_ERROR();
    }

    TIME_PAINTGL_DETAILS ComputationCheckError();
    TIME_PAINTGL_DETAILS GlException_CHECK_ERROR();

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    TIME_PAINTGL_DRAW
    {
        unsigned collections_n = 0;
        for (i=0; i < N; ++i)
            collections_n += collections[i].read ()->isVisible();

        TaskInfo("Drew %u channels*%u block%s*%u triangles (%u triangles in total) in viewport(%d, %d).",
        collections_n,
        model->renderer->render_settings.drawn_blocks,
        model->renderer->render_settings.drawn_blocks==1?"":"s",
        model->renderer->trianglesPerBlock(),
        collections_n*model->renderer->render_settings.drawn_blocks*model->renderer->trianglesPerBlock(),
        current_viewport[2], current_viewport[3]);
    }
}


void RenderView::
        drawCollection(int i, float yscale )
{
    model->renderer->collection = model->collections()[i];
    model->renderer->render_settings.fixed_color = channel_colors[i];
    glDisable(GL_BLEND);
    if (0 != model->_rx)
        glEnable( GL_CULL_FACE ); // enabled only while drawing collections
    else
        glEnable( GL_DEPTH_TEST );
    model->renderer->draw( yscale, this->last_length ()); // 0.6 ms
    glDisable( GL_CULL_FACE );
    glEnable(GL_BLEND);
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


    // Must disable texturing and lighting as well when drawing primitives
    glDisable(GL_COLOR_MATERIAL);
    //glEnable(GL_TEXTURE_2D);
    //glEnable(GL_NORMALIZE);

    setLights();

    //float materialSpecular[] = {0.5f, 0.5f, 0.5f, 1.0f};
    //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, materialSpecular);
    //glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0f);

    GlException_CHECK_ERROR();
}


void RenderView::
        setLights()
{
    glDisable(GL_LIGHTING);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

/*    GLfloat LightAmbient[]= { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };
    //GLfloat LightDirection[]= { 0.0f, 0.0f, 1.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, LightDiffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);
    //glLightfv(GL_LIGHT0, GL_POSITION, LightDirection);
    glEnable(GL_LIGHT0);*/
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
        setPosition( Heightmap::Position pos )
{
    float l = model->project()->length();
    model->_qx = pos.time;
    if (model->_qx<0) model->_qx=0;
    if (model->_qx>l) model->_qx=l;

    model->_qz = pos.scale;
    if (model->_qz<0) model->_qz=0;
    if (model->_qz>1) model->_qz=1;

    redraw();
}


void RenderView::
        setLastUpdateSize( Signal::UnsignedIntervalType last_update_size )
{
    // _last_update_size must be non-zero to be divisable
    _last_update_size = std::max(1llu, last_update_size);

    if ((Signal::UnsignedIntervalType)Signal::Interval::IntervalType_MAX < _last_update_size)
      {
        // Oddly enough
        // '_last_update_size' is close but not equal to 'Signal::Interval::Interval_ALL.count ()'
      }
}


Support::ToolSelector* RenderView::
        toolSelector()
{
//    if (!tool_selector_)
//        tool_selector_.reset( new Support::ToolSelector(glwidget));

    return tool_selector;
}


float RenderView::
        last_length()
{
    return model->project()->length();
}


void RenderView::
        emitTransformChanged()
{
    channel_colors.clear ();
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
    emit postUpdate();
}


void RenderView::
        restartUpdateTimer()
{
    if (_update_timer->isActive())
        return;

    float dt = _last_frame.elapsed();
    float wait = 1.f/60.f - 0.0015f; // 1.5 ms overhead

    // Sleeping in _update_timer is not needed if vsync is in use
    if (const QGLContext* context = QGLContext::currentContext ())
      {
        bool vsync = 0 < context->format ().swapInterval ();
        if (vsync)
            wait = 0;
      }

    if (wait < dt)
        wait = dt;

    unsigned ms = (wait-dt)*1e3; // round down
    if (ms < 3) // but don't stall
        ms = 3;

    _update_timer->start(ms);
}


void RenderView::
        initializeGL()
{
    _inited = true;

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


void RenderView::
        paintGL()
{
    model->renderer->collection = model->tfr_mapping ().read ()->collections()[0];
    model->renderer->init();
    if (!model->renderer->isInitialized())
        return;

    float elapsed_ms = -1;

    TIME_PAINTGL_DETAILS if (_render_timer)
	    elapsed_ms = _render_timer->elapsedTime()*1000.f;
    TIME_PAINTGL_DETAILS _render_timer.reset();
    TIME_PAINTGL_DETAILS _render_timer.reset(new TaskTimer("Time since last RenderView::paintGL (%g ms, %g fps)", elapsed_ms, 1000.f/elapsed_ms));

    TIME_PAINTGL TaskTimer tt("............................. RenderView::paintGL.............................");

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

        if(0) foreach( const Heightmap::Collection::Ptr& c, collections )
        {
            c.read ()->printCacheSize();
        }
    }


    _try_gc = 0;
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

    bool chunk_merger_has_work = !model->chunk_merger->processChunks(0);
    //model->chunk_merger->processChunks(-1);

    if (chunk_merger_has_work)
        redraw (); // won't redraw right away, but enqueue an update


    // Set up camera position
    {
        TIME_PAINTGL_DETAILS TaskTimer tt("Set up camera position");

        setupCamera();

        glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
        glGetIntegerv(GL_VIEWPORT, viewport_matrix);

        // drawCollections shouldn't use the matrix applied by setRotationForAxes
        glPushMatrixContext ctx(GL_MODELVIEW);
        setRotationForAxes(false);
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
    }

    bool onlyComputeBlocksForRenderView = false;
    Signal::OperationDesc::Extent x;
    { // Render
		TIME_PAINTGL_DETAILS TaskTimer tt("Render");
        float length=0.f;

        if (onlyComputeBlocksForRenderView)
        foreach( const Heightmap::Collection::Ptr& collection, collections )
        {
            collection.write ()->next_frame(); // Discard needed blocks before this row
        }

        Signal::Processing::Step::Ptr step_with_new_extent;
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

        drawCollections( _renderview_fbo.get(), model->_rx>=45 ? 1 - model->orthoview : 1 );

        last_ysize = model->renderer->render_settings.last_ysize;
        glScalef(1, last_ysize*1.5<1.?last_ysize*1.5:1., 1); // global effect on all tools

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
            model->renderer->gl_projection.update ();
//            memcpy( model->renderer->viewport_matrix, viewport_matrix, sizeof(viewport_matrix));
//            memcpy( model->renderer->modelview_matrix, modelview_matrix, sizeof(modelview_matrix));
//            memcpy( model->renderer->projection_matrix, projection_matrix, sizeof(projection_matrix));

            model->renderer->drawAxes( length ); // 4.7 ms

            model->renderer->render_settings.draw_piano = draw_piano;
            model->renderer->render_settings.draw_hz = draw_hz;
            model->renderer->render_settings.draw_t = draw_t;
        }
    }


    // It should update the view in sections with the same size as blocks
    Signal::Processing::TargetNeeds::Ptr target_needs = model->target_marker()->target_needs();
    Support::HeightmapProcessingPublisher wu(target_needs, model->collections());
    wu.update(model->_qx, x, _last_update_size);

    Support::ChainInfo ci(model->project ()->processing_chain ());
    bool isWorking = ci.hasWork () || chunk_merger_has_work;
    int n_workers = ci.n_workers ();
    int dead_workers = ci.dead_workers ();
    if (wu.failedAllocation ())
        dead_workers += n_workers;
    // dead_workers = (wu.failedAllocation () || n_workers==0) && !isRecording

    if (isWorking || isRecording || dead_workers) {
        Support::DrawWorking::drawWorking( viewport_matrix[2], viewport_matrix[3], n_workers, dead_workers );
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
        foreach( const Heightmap::Collection::Ptr& collection, collections )
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

    _try_gc = 0;

#ifdef USE_CUDA
    } catch (const CudaException &x) {
        TaskInfo tt("RenderView::paintGL CAUGHT CUDAEXCEPTION\n%s", x.what());

        if (2>_try_gc)
        {
            Sawe::Application::global_ptr()->clearCaches();

            if (cudaErrorMemoryAllocation == x.getCudaError() && _try_gc == 0)
            {
                Tfr::Cwt* cwt = model->getParam<Tfr::Cwt>();
                TaskInfo("scales_per_octave was %g", cwt->scales_per_octave());

                float fs = worker.target()->source()->sample_rate();
                cwt->scales_per_octave( cwt->scales_per_octave() , fs );

                TaskInfo("scales_per_octave is %g", cwt->scales_per_octave());

                model->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );
            }
            emit transformChanged();

            _try_gc++;
        }
        else throw;
#endif
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
    foreach( const Heightmap::Collection::Ptr& collection, model->collections() )
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


void RenderView::
        computeChannelColors()
{
    unsigned N = model->collections().size();
    channel_colors.resize( N );

    // Set colors
    float R = 0, G = 0, B = 0;
    for (unsigned i=0; i<N; ++i)
    {
        QColor c = QColor::fromHsvF( i/(float)N, 1, 1 );
        channel_colors[i] = tvector<4>(c.redF(), c.greenF(), c.blueF(), c.alphaF());
        R += channel_colors[i][0];
        G += channel_colors[i][1];
        B += channel_colors[i][2];
    }

    // R, G and B sum up to the same constant = N/2 if N > 1
    for (unsigned i=0; i<N; ++i)
    {
        channel_colors[i] = channel_colors[i] * (N/2.f);
    }

    if(0) if (1==N) // There is a grayscale mode to use for this
        channel_colors[0] = tvector<4>(0,0,0,1);
}


} // namespace Tools
