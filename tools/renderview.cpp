// gl
#include <gl.h>

#include "renderview.h"

// TODO cleanup
#include "ui/mainwindow.h"

// Sonic AWE
#include "sawe/project.h"
#include "tfr/cwt.h"
#include "toolfactory.h"
#include "support/drawworking.h"
#include "adapters/microphonerecorder.h"
#include "heightmap/renderer.h"
#include "sawe/application.h"

// gpumisc
#include <CudaException.h>
#include <GlException.h>
#include <glPushContext.h>
#include <demangle.h>
#include <cuda_vector_types_op.h>
#include <glframebuffer.h>

// cuda
#include <cuda.h>

// Qt
#include <QTimer>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>
#include <QGLContext>
#include <QGraphicsView>


#define TIME_PAINTGL
//#define TIME_PAINTGL if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

//#define DEBUG_EVENTS
#define DEBUG_EVENTS if(0)

#ifdef max
#undef max
#endif

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            last_ysize(1),
            orthoview(1),
            model(model),
            glwidget(0),
            graphicsview(0),
            _work_timer( new TaskTimer("Benchmarking first work")),
            _inited(false),
            _last_width(0),
            _last_height(0),
            _last_x(0),
            _last_y(0),
            _try_gc(0)
{
    float l = model->project()->worker.source()->length();
    _last_length = l;

    // Validate rotation and set orthoview accordingly
    if (model->_rx<0) model->_rx=0;
    if (model->_rx>=90) { model->_rx=90; orthoview.reset(1); } else orthoview.reset(0);
    //if (0<orthoview && model->_rx<90) { model->_rx=90; orthoview=0; }

    computeChannelColors();

    connect( Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), SLOT(clearCaches()) );
    connect( this, SIGNAL(finishedWorkSection()), SLOT(finishedWorkSectionSlot()), Qt::QueuedConnection );
    connect( this, SIGNAL(sceneRectChanged ( const QRectF & )), SLOT(userinput_update()) );

    connect( this, SIGNAL(postUpdate()), SLOT(update()), Qt::QueuedConnection );
}


RenderView::
        ~RenderView()
{
    TaskTimer tt("%s", __FUNCTION__);

    glwidget->makeCurrent();

    emit destroying();

    _render_timer.reset();
    _work_timer.reset();
    _renderview_fbo.reset();

    QGraphicsScene::clear();

    if ( 1 < Sawe::Application::global_ptr()->count_projects())
        return;

    Sawe::Application::global_ptr()->clearCaches();

    TaskInfo("cudaThreadExit()");

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
    glwidget->makeCurrent();

    BOOST_ASSERT( QGLContext::currentContext() );

    // Destroy the cuda context for this thread
    CudaException_SAFE_CALL( cudaThreadExit() );
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
    painter->beginNativePainting();

    glMatrixMode(GL_MODELVIEW);

    {
        glPushAttribContext attribs;
        glPushMatrixContext pmcp(GL_PROJECTION);
        glPushMatrixContext pmcm(GL_MODELVIEW);

        if (!_inited)
            initializeGL();

		if (painter->device())
		{
            unsigned w = painter->device()->width();
            unsigned h = painter->device()->height();
            if (w != _last_width || h != _last_height)
                userinput_update();
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

        resizeGL(_last_width, _last_height );

        paintGL();

        {
            glScalef(1,1,0.1f);
            glRotatef(90,1,0,0);
            GLdouble m[16];//, proj[16];
            GLint vp[4];
            glGetDoublev(GL_MODELVIEW_MATRIX, m);
//            glGetDoublev(GL_PROJECTION_MATRIX, proj);
            glGetIntegerv(GL_VIEWPORT, vp);



//            projectionTransform.setMatrix( proj[0], proj[1], proj[2],
//                                           proj[4], proj[5], proj[6],
//                                           proj[8], proj[9], proj[10]);
            /*
             This would make a mapping from 3D to the 2D plane.
            if (qFuzzyCompare(m[3] + 1, 1) && qFuzzyCompare(m[7] + 1, 1))
            {
                modelviewTransform = QTransform(m[0]/m[15], m[1]/m[15], m[4]/m[15],
                                                m[5]/m[15], m[12]/m[15], m[13]/m[15]);
            }
            else
                modelviewTransform = QTransform(m[0], m[1], m[3],
                                                m[4], m[5], m[7],
                                                m[12], m[13], m[15]);

            viewTransform = QTransform(vp[2]*0.5, 0,
                                       0, -vp[3]*0.5,
                                      vp[0]+vp[2]*0.5, vp[1]+vp[3]*0.5);
            */
        }

        defaultStates();
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
#ifdef __APPLE__
    return 0; //Crash on mac os
#endif
#ifdef _MSC_VER
    // getHeightmapValue is tremendously slow in windows for some reason
    //return 0;
#endif
    if (is_valid_value)
        *is_valid_value = true;

    if (pos.time < 0 || pos.scale < 0 || pos.scale > 1 || pos.time > _last_length)
        return 0;

    if (is_valid_value)
        *is_valid_value = false;

    Heightmap::Reference myref(model->collections[0].get());
    if (!ref)
    {
        ref = &myref;
        ref->block_index[0] = (unsigned)-1;
    }
    if (ref->block_index[0] == (unsigned)-1)
    {
        *ref = findRefAtCurrentZoomLevel( pos );
    }

    Heightmap::Position a,b;
    ref->getArea( a, b );

    ref->block_index[0] = pos.time / (b.time - a.time);
    ref->block_index[1] = pos.scale / (b.scale - a.scale);
    ref->getArea( a, b );

    Heightmap::pBlock block = model->collections[0]->getBlock( *ref );
    if (!block)
        return 0;
    if (is_valid_value)
    {
        Signal::IntervalType s = pos.time * ref->sample_rate();
        Signal::Intervals I = Signal::Intervals(s, s+1);
        I -= block->valid_samples;
        I &= ref->getInterval();
        if (I.empty())
        {
            *is_valid_value = true;
        } else
            return 0;
    }

    GpuCpuData<float>* blockData = block->glblock->height()->data.get();

    float* data = blockData->getCpuMemory();
    unsigned w = ref->samplesPerBlock();
    unsigned h = ref->scalesPerBlock();
    unsigned x0 = (pos.time-a.time)/(b.time-a.time)*(w-1) + .5f;
    float    yf = (pos.scale-a.scale)/(b.scale-a.scale)*(h-1);
    unsigned y0 = yf + .5f;

    BOOST_ASSERT( x0 < w );
    BOOST_ASSERT( y0 < h );
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
                *pick_local_max = a.scale + (b.scale-a.scale)*(y0 + m0)/(h-1);
        }
        else
        {
            v = v2;
            if (pick_local_max)
                *pick_local_max = a.scale + (b.scale-a.scale)*(y0)/(h-1);
        }
    }

    return v;
}


Heightmap::Reference RenderView::
        findRefAtCurrentZoomLevel(Heightmap::Position p)
{
    memcpy( model->renderer->viewport_matrix, viewport_matrix, sizeof(viewport_matrix));
    memcpy( model->renderer->modelview_matrix, modelview_matrix, sizeof(modelview_matrix));
    memcpy( model->renderer->projection_matrix, projection_matrix, sizeof(projection_matrix));

    model->renderer->collection = model->collections[0].get();

    return model->renderer->findRefAtCurrentZoomLevel( p );
}


QPointF RenderView::
        getScreenPos( Heightmap::Position pos, double* dist, bool use_heightmap_value )
{
    GLdouble objY = 0;
    if (1 != orthoview && use_heightmap_value)
        objY = getHeightmapValue(pos) * model->renderer->y_scale * 4 * last_ysize;

    GLdouble winX, winY, winZ;
    gluProject( pos.time, objY, pos.scale,
                modelview_matrix, projection_matrix, viewport_matrix,
                &winX, &winY, &winZ);

    if (dist)
    {
        GLint const* const& vp = viewport_matrix;
        float z0 = .1, z1=.2;
        GLvector projectionPlane = Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z0), modelview_matrix, projection_matrix, vp );
        GLvector projectionNormal = (Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z1), modelview_matrix, projection_matrix, vp ) - projectionPlane);

        GLvector p;
        p[0] = pos.time;
        p[1] = 0;//objY;
        p[2] = pos.scale;

        GLvector d = p-projectionPlane;
        projectionNormal[0] *= model->xscale;
        projectionNormal[2] *= last_ysize;
        d[0] *= model->xscale;
        d[2] *= last_ysize;

        projectionNormal.Normalize();
        *dist = d%projectionNormal;
    }

    return QPointF( winX, _last_height-1 - winY );
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
    if (1 == orthoview)
        return getPlanePos(widget_pos, 0, useRenderViewContext);

    TaskTimer tt("RenderView::getHeightmapPos(%g, %g) Newton raphson", widget_pos.x(), widget_pos.y());

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

        y = getHeightmapValue(p) * model->renderer->y_scale * 4 * last_ysize;
        tt.info("(%g, %g) %g is Screen(%g, %g), s = %g", p.time, p.scale, y, r.x(), r.y(), s);
        prevs = s;
    }

    y = getHeightmapValue(p) * model->renderer->y_scale * 4 * last_ysize;
    TaskInfo("Screen(%g, %g) projects at Heightmap(%g, %g, %g)", widget_pos.x(), widget_pos.y(), p.time, p.scale, y);
    QPointF r = getWidgetPos( p, 0 );
    TaskInfo("Heightmap(%g, %g) projects at Screen(%g, %g)", p.time, p.scale, r.x(), r.y() );
    return p;
}

Heightmap::Position RenderView::
        getPlanePos( QPointF pos, bool* success, bool useRenderViewContext )
{
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

    float ylevel = 0;
    float s = (ylevel-objY1)/(objY2-objY1);

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
    GlException_CHECK_ERROR();

    unsigned N = model->collections.size();

    TIME_PAINTGL_DETAILS CudaException_CHECK_ERROR();

    // Draw the first channel without a frame buffer
    model->renderer->camera = GLvector(model->_qx, model->_qy, model->_qz);

    // When rendering to fbo, draw to the entire fbo, then update the current
    // viewport.
    GLint current_viewport[4];
    glGetIntegerv(GL_VIEWPORT, current_viewport);

    TIME_PAINTGL_DETAILS TaskTimer tt("Viewport (%u, %u, %u, %u)",
        current_viewport[0], current_viewport[1],
        current_viewport[2], current_viewport[3]);

    for (unsigned i=0; i < 1; ++i)
        drawCollection(i, yscale);

    if (1<N)
    {
        // drawCollections is called for 3 different viewports each frame.
        // GlFrameBuffer will query the current viewport to determine the size
        // of the fbo for this iteration.
        fbo->recreate();

        // Could also recreate the fbo each frame
        // boost::scoped_ptr<GlFrameBuffer> my_fbo;
        // if (!fbo)
        // {
        //     my_fbo.reset( new GlFrameBuffer );
        //     fbo = my_fbo.get();
        // }

        for (unsigned i=1; i < N; ++i)
        {
            GlException_CHECK_ERROR();

            {
                GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding();
                glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
                glViewport(0, 0,
                           fbo->getGlTexture().getWidth(), fbo->getGlTexture().getHeight());

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
            GlTexture::ScopeBinding texObjBinding = fbo->getGlTexture().getScopeBinding();
            glBegin(GL_TRIANGLE_STRIP);
                glTexCoord2f(0,0); glVertex2f(0,0);
                glTexCoord2f(1,0); glVertex2f(1,0);
                glTexCoord2f(0,1); glVertex2f(0,1);
                glTexCoord2f(1,1); glVertex2f(1,1);
            glEnd();

            glEnable(GL_DEPTH_TEST);

            GlException_CHECK_ERROR();
        }
    }

    TIME_PAINTGL_DETAILS CudaException_CHECK_ERROR();
    TIME_PAINTGL_DETAILS GlException_CHECK_ERROR();

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    TIME_PAINTGL_DETAILS TaskInfo("Drew %u*%u block%s",
        N,
        model->renderer->drawn_blocks, 
        model->renderer->drawn_blocks==1?"":"s");
}


void RenderView::
        drawCollection(int i, float yscale )
{
    model->renderSignalTarget->source()->set_channel( i );
    model->renderer->collection = model->collections[i].get();
    model->renderer->fixed_color = channel_colors[i];
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glEnable( GL_CULL_FACE ); // enabled only while drawing collections
    model->renderer->draw( yscale ); // 0.6 ms
    glDisable( GL_CULL_FACE );
    glDisable(GL_LIGHTING);
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

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glFrontFace( model->renderer->left_handed_axes ? GL_CCW : GL_CW );
    glCullFace( GL_BACK );
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
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
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    GLfloat LightAmbient[]= { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };
    //GLfloat LightDirection[]= { 0.0f, 0.0f, 1.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, LightDiffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);
    //glLightfv(GL_LIGHT0, GL_POSITION, LightDirection);
    glEnable(GL_LIGHT0);
}


void RenderView::
        defaultStates()
{
    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    //glDisable(GL_COLOR_MATERIAL);
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
    model->_qx = pos.time;
    if (model->_qx<0) model->_qx=0;
    if (model->_qx>_last_length) model->_qx=_last_length;

    model->_qz = pos.scale;
    if (model->_qz<0) model->_qz=0;
    if (model->_qz>1) model->_qz=1;

    userinput_update();
}


Support::ToolSelector* RenderView::
        toolSelector()
{
//    if (!tool_selector_)
//        tool_selector_.reset( new Support::ToolSelector(glwidget));

    return tool_selector;
}


void RenderView::
        userinput_update( bool request_high_fps )
{
    if (request_high_fps)
        model->project()->worker.requested_fps(60);

    emit postUpdate();
}


void RenderView::
        initializeGL()
{
    //printQGLWidget(*this, "this");
    //TaskTimer("autoBufferSwap=%d", autoBufferSwap()).suppressTiming();
    _inited = true;

    TaskInfo("_renderview_fbo");
    if (!_renderview_fbo) _renderview_fbo.reset( new GlFrameBuffer );
}


void RenderView::
        resizeGL( int width, int height )
{
    height = height?height:1;

    QRect rect = tool_selector->parentTool()->geometry();
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
    if (!model->renderSignalTarget)
        return;

    float elapsed_ms = -1;

    TIME_PAINTGL_DETAILS if (_render_timer)
	    elapsed_ms = _render_timer->elapsedTime()*1000.f;
    TIME_PAINTGL_DETAILS _render_timer.reset();
    TIME_PAINTGL_DETAILS _render_timer.reset(new TaskTimer("Time since last RenderView::paintGL (%g ms, %g fps)", elapsed_ms, 1000.f/elapsed_ms));

    Signal::Worker& worker = model->project()->worker;
    Signal::Operation* first_source = worker.source()->root();

    TIME_PAINTGL TaskTimer tt("............................. RenderView::paintGL %s (%p).............................",
                              first_source->name().c_str(), first_source);

    unsigned N = model->collections.size();
    unsigned long sumsize = 0;
    unsigned cacheCount = 0;
    TIME_PAINTGL_DETAILS
    {
        sumsize = model->collections[0]->cacheByteSize();
        cacheCount = model->collections[0]->cacheCount();
        for (unsigned i=1; i<N; ++i)
        {
            TaskLogIfFalse( sumsize == model->collections[i]->cacheByteSize() );
            TaskLogIfFalse( cacheCount == model->collections[i]->cacheCount() );
        }
    }

    TIME_PAINTGL_DETAILS TaskInfo("Drawing (%g MB cache for %u*%u blocks) of %s (%p) %s",
        N*sumsize/1024.f/1024.f, N, cacheCount, vartype(*first_source).c_str(), first_source, first_source->name().c_str());
    if(0) TIME_PAINTGL_DETAILS for (unsigned i=0; i<N; ++i)
    {
        model->collections[i]->printCacheSize();
    }


    _try_gc = 0;
    try {
		{
			TIME_PAINTGL_DETAILS TaskTimer tt("paintGL pre check errors");
			GlException_CHECK_ERROR();
		}
		{
			TIME_PAINTGL_DETAILS TaskTimer tt("paintGL pre sync");
            CudaException_ThreadSynchronize();
		}

        {
            // Make sure our cuda context is still alive by invoking
            // a tiny kernel. This will throw an CudaException otherwise,
            // thus resulting in restarting the cuda context.

            // If we don't ensure the context is alive before starting to
            // use various GPU resources the application will crash, for
            // instance when another RenderView is closed and releases
            // the context.
            Tfr::Stft a;
            a.set_approximate_chunk_size(4);
            Signal::pBuffer b(new Signal::Buffer(0,a.chunk_size(),1));
            a(b);
        }


    // Set up camera position
    _last_length = model->renderSignalTarget->source()->length();
    {   
		TIME_PAINTGL_DETAILS TaskTimer tt("Set up camera position");

        setupCamera();

        glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
        glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
        glGetIntegerv(GL_VIEWPORT, viewport_matrix);
	}

    // TODO move to rendercontroller
    bool isWorking = false;
    bool isRecording = false;

    Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );
    if(r != 0 && !(r->isStopped()))
        isRecording = true;

    bool onlyComputeBlocksForRenderView = false;
    { // Render
		TIME_PAINTGL_DETAILS TaskTimer tt("Render");

        if (onlyComputeBlocksForRenderView)
        foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
        {
            collection->next_frame(); // Discard needed blocks before this row
        }

        drawCollections( _renderview_fbo.get(), 1 - orthoview );

        last_ysize = model->renderer->last_ysize;
        glScalef(1, last_ysize*1.5<1.?last_ysize*1.5:1., 1); // global effect on all tools

		{
			TIME_PAINTGL_DETAILS TaskTimer tt("emit painting");
			emit painting();
		}

        model->renderer->drawAxes( _last_length ); // 4.7 ms

    }

    {   // Find things to work on (ie playback and file output)
		TIME_PAINTGL_DETAILS TaskTimer tt("Find things to work on");

        worker.center = model->_qx;

        emit populateTodoList();

        if (!worker.target()->post_sink()->isUnderfed())
        {
            // the todo list in worker isn't updated unless Worker::target(pTarget) is called.
            worker.target( model->renderSignalTarget );
        }
    }

    {   // Work
        isWorking = worker.todo_list();
        TIME_PAINTGL_DETAILS TaskTimer tt("Work target = %s, todo list = %s, isWorking = %d",
                 worker.target()->name().c_str(),
                 worker.todo_list().toString().c_str(), isWorking);

        if (isWorking || isRecording) {
            if (!_work_timer.get())
                _work_timer.reset( new TaskTimer("Working"));

            // project->worker can be run in one or more separate threads, but if it isn't
            // execute the computations for one chunk
#ifndef SAWE_NO_MUTEX
            if (!worker.isRunning()) {
                worker.workOne(!isRecording);
                emit postUpdate();
            } else {
                //project->worker.todo_list().print("Work to do");
                // Wait a bit while the other thread work
                QTimer::singleShot(200, this, SLOT(update()));

                worker.checkForErrors();
            }
#else
            worker.workOne(!isRecording);
            emit postUpdate();
#endif
        } else {
            static unsigned workcount = 0;
            if (_work_timer) {
                float worked_time = worker.worked_samples.count()/worker.source()->sample_rate();
                _work_timer->info("Finished %u chunks covering %g s (%g x realtime). Work session #%u",
                                  worker.work_chunks,
                                  worked_time,
                                  worked_time/_work_timer->elapsedTime(),
                                  workcount);
                worker.work_chunks = 0;
                worker.worked_samples.clear();
                workcount++;
                _work_timer.reset();

                // Useful when debugging to close application or do something else after finishing first work chunk
                emit finishedWorkSection();
            }
        }
    }

    if (isWorking || isRecording)
        Support::DrawWorking::drawWorking( viewport_matrix[2], viewport_matrix[3] );

    if (!onlyComputeBlocksForRenderView)
	{
		TIME_PAINTGL_DETAILS TaskTimer tt("collection->next_frame");
		foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
		{
	        // Start looking for which blocks that are requested for the next frame.
			collection->next_frame();
		}
	}

		{
			TIME_PAINTGL_DETAILS TaskTimer tt("paintGL post check errors");
    GlException_CHECK_ERROR();
		}
		{
			TIME_PAINTGL_DETAILS TaskTimer tt("paintGL post sync");
			CudaException_CHECK_ERROR();
		}

    _try_gc = 0;
    } catch (const CudaException &x) {
        TaskInfo tt("RenderView::paintGL CAUGHT CUDAEXCEPTION\n%s", x.what());

        float scales_per_octave = Tfr::Cwt::Singleton().scales_per_octave();

        if (2>_try_gc)
        {
            Sawe::Application::global_ptr()->clearCaches();

            if (cudaErrorMemoryAllocation == x.getCudaError() && _try_gc == 0)
            {
                TaskInfo("scales_per_octave was %g", Tfr::Cwt::Singleton().scales_per_octave());

                size_t free=0, total=0;
                cudaMemGetInfo(&free, &total);

                float fs = worker.target()->source()->sample_rate();
                Tfr::Cwt::Singleton().scales_per_octave( scales_per_octave / 0.99 );
                unsigned L;
                do
                {
                    Tfr::Cwt::Singleton().scales_per_octave( Tfr::Cwt::Singleton().scales_per_octave() * .99 );
                    if (Tfr::Cwt::Singleton().scales_per_octave() < 1)
                    {
                        Tfr::Cwt::Singleton().scales_per_octave( 1 );
                        break;
                    }

                    L = Tfr::Cwt::Singleton().next_good_size(1, fs);
                }
                while (free < Tfr::Cwt::Singleton().required_gpu_bytes(L, fs ) );

                TaskInfo("scales_per_octave is %g", Tfr::Cwt::Singleton().scales_per_octave());
            }
            emit transformChanged();

            _try_gc++;
        }
        else throw;
    } catch (const GlException &x) {
        TaskTimer tt("RenderView::paintGL CAUGHT GLEXCEPTION\n%s", x.what());
        if (2>_try_gc) {
            Sawe::Application::global_ptr()->clearCaches();
            emit transformChanged();
            _try_gc++;
        }
        else throw;
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
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
    {
        Heightmap::Collection* c = collection.get();
        c->reset(); // note, not c.reset()
    }

    Heightmap::Renderer::ColorMode old_color_mode = model->renderer->color_mode;
    model->renderer.reset();
    model->renderer.reset(new Heightmap::Renderer( model->collections[0].get() ));
    model->renderer->color_mode = old_color_mode;

    userinput_update();
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
    if (model->_rx<90) orthoview = 0;
    if (orthoview != 1 && orthoview != 0)
        userinput_update();

    glLoadIdentity();
    glTranslatef( model->_px, model->_py, model->_pz );

    glRotatef( model->_rx, 1, 0, 0 );
    glRotatef( fmod(fmod(model->_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(model->_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( model->_rz, 0, 0, 1 );

    if (model->renderer->left_handed_axes)
        glScalef(-1, 1, 1);
    else
    {
        glRotatef(-90,0,1,0);
        glScalef(0.35, 1, 2.6);
    }

    glScalef(model->xscale, 1, model->zscale);

    glTranslatef( -model->_qx, -model->_qy, -model->_qz );

    orthoview.TimeStep(.08);
}


void RenderView::
        computeChannelColors()
{
    unsigned N = model->collections.size();
    channel_colors.resize( N );

    // Set colors
    float R = 0, G = 0, B = 0;
    for (unsigned i=0; i<N; ++i)
    {
        QColor c = QColor::fromHsvF( i/(float)N, 1, 1 );
        channel_colors[i] = make_float4(c.redF(), c.greenF(), c.blueF(), c.alphaF());
        R += channel_colors[i].x;
        G += channel_colors[i].y;
        B += channel_colors[i].z;
    }

    // R, G and B sum up to the same constant = N/2 if N > 1
    for (unsigned i=0; i<N; ++i)
    {
        channel_colors[i] = channel_colors[i] * (N/2.f);
    }

    if(0) if (1==N) // There is a grayscale mode to use for this
        channel_colors[0] = make_float4(0,0,0,1);
}


} // namespace Tools


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
