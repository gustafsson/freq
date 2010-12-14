// gl
#include "GL/glew.h"

#include "renderview.h"

// TODO cleanup

// Sonic AWE
#include "sawe/project.h"
#include "tfr/cwt.h"
#include "toolfactory.h"
#include "support/drawworking.h"
#include "adapters/microphonerecorder.h"
#include "heightmap/renderer.h"

// gpumisc
#include <CudaException.h>
#include <GlException.h>
#include <glPushContext.h>
#include <demangle.h>
#include <cuda_vector_types_op.h>
#include <glframebuffer.h>

// Qt
#include <QTimer>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

//#define DEBUG_EVENTS
#define DEBUG_EVENTS if(0)

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            //QGLWidget(QGLFormat(QGL::SampleBuffers)),
            orthoview(1),
            model(model),
            _work_timer( new TaskTimer("Benchmarking first work")),
            _inited(false)
{
    float l = model->project()->worker.source()->length();
    _prevLimit = l;
    _last_length = l;

    // Validate rotation and set orthoview accordingly
    if (model->_rx<0) model->_rx=0;
    if (model->_rx>=90) { model->_rx=90; orthoview.reset(1); } else orthoview.reset(0);
    //if (0<orthoview && model->_rx<90) { model->_rx=90; orthoview=0; }

    computeChannelColors();
}


RenderView::
        ~RenderView()
{
    TaskTimer tt("%s, calling cudaThreadExit()", __FUNCTION__);

    emit destroying();

    // Because the Cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
    //
    // Note though that other OpenGL contexts might still be active in other
    // Sonic AWE windows. The Cuda context would probably only need to be
    // destroyed prior to the destruction of the last OpenGL context. This
    // would require further investigation.
    makeCurrent();

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
			_last_width = painter->device()->width();
			_last_height = painter->device()->height();
		}

        setStates();
        resizeGL(_last_width, _last_height);

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
        }

        defaultStates();
    }

    painter->endNativePainting();
}


float RenderView::
        getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref )
{
#ifdef WIN32
    // getHeightmapValue is tremendously slow on windos for some reason
    return 0;
#endif

    if (pos.time < 0 || pos.scale < 0 || pos.scale > 1 || pos.time > _last_length)
        return 0;

    memcpy( model->renderer->viewport_matrix, viewport_matrix, sizeof(viewport_matrix));
    memcpy( model->renderer->modelview_matrix, modelview_matrix, sizeof(modelview_matrix));
    memcpy( model->renderer->projection_matrix, projection_matrix, sizeof(projection_matrix));

    Heightmap::Reference myref(model->collections[0].get());
    if (!ref)
    {
        ref = &myref;
        ref->block_index[0] = (unsigned)-1;
    }
    if (ref->block_index[0] == (unsigned)-1)
        *ref = model->renderer->findRefAtCurrentZoomLevel( pos.time, pos.scale );

    Heightmap::Position a,b;
    ref->getArea( a, b );

    ref->block_index[0] = pos.time / (b.time - a.time);
    ref->block_index[1] = pos.scale / (b.scale - a.scale);
    ref->getArea( a, b );

    Heightmap::pBlock block = model->collections[0]->getBlock( *ref );
    if (!block)
        return 0;
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();

    float* data = blockData->getCpuMemory();
    unsigned w = ref->samplesPerBlock();
    unsigned h = ref->scalesPerBlock();
    unsigned x0 = (pos.time-a.time)/(b.time-a.time)*(w-1) + .5f;
    unsigned y0 = (pos.scale-a.scale)/(b.scale-a.scale)*(h-1) + .5f;

    BOOST_ASSERT( x0 < w );
    BOOST_ASSERT( y0 < h );
    float v = data[ x0 + y0*w ];

    v *= model->renderer->y_scale;
    v *= 4;
    return v;
}


QPointF RenderView::
        getScreenPos( Heightmap::Position pos, double* dist )
{
    GLdouble objY;
    if (1 != orthoview)
        objY = getHeightmapValue(pos) * last_ysize;

    GLdouble winX, winY, winZ;
    gluProject( pos.time, objY, pos.scale,
                modelview_matrix, projection_matrix, viewport_matrix,
                &winX, &winY, &winZ);

    if (dist)
    {
        GLint const* const& vp = viewport_matrix;
        float z0 = .1, z1=.2;
        GLvector projectionPlane = Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z0), modelview_matrix, projection_matrix, vp );
        GLvector projectionNormal = (Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z1), modelview_matrix, projection_matrix, vp ) - projectionPlane).Normalize();

        GLvector p;
        p[0] = pos.time;
        p[1] = 0;//objY;
        p[2] = pos.scale;

        *dist = (p-projectionPlane)%projectionNormal;
        *dist *= last_ysize;
    }

    return QPointF( winX, viewport_matrix[3]-1-winY );
    //return QPointF( winX, winY );
}


Heightmap::Position RenderView::
        getHeightmapPos( QPointF pos, bool useRenderViewContext )
{
    if (1 == orthoview)
        return getPlanePos(pos, 0, useRenderViewContext);

    TaskTimer tt("RenderView::getPlanePos Newton raphson");

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
    float y = 0;

    // Newton raphson
    for (int i=0; i<10; ++i)
    {
        float s = (y-objY1)/(objY2-objY1);

        Heightmap::Position q;
        q.time = objX1 + s * (objX2-objX1);
        q.scale = objZ1 + s * (objZ2-objZ1);

        float e = (q.time-p.time)*(q.time-p.time)*model->xscale*model->xscale + (q.scale-p.scale)*(q.scale-p.scale);
        p = q;
        if (e < 1e-5 )
            break;

        y = getHeightmapValue(p) * last_ysize;
        tt.info("(%g, %g) %g", p.time, p.scale, y);
    }
    return p;
}

Heightmap::Position RenderView::
        getPlanePos( QPointF pos, bool* success, bool useRenderViewContext )
{
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
    if (0) if(success)
    {
        if( s < 0)
            if (success) *success=false;

        float L = sqrt((objX1-objX2)*(objX1-objX2)
                       +(objY1-objY2)*(objY1-objY2)
                       +(objZ1-objZ2)*(objZ1-objZ2));
        if (objY1-objY2 < model->xscale*sin(minAngle *(M_PI/180)) * L )
            if (success) *success=false;
    }

    return p;
}


void RenderView::
        drawCollections()
{
    GlException_CHECK_ERROR();

    unsigned N = model->collections.size();
    unsigned long sumsize = 0;
    TIME_PAINTGL for (unsigned i=0; i<N; ++i)
        sumsize += model->collections[i]->cacheByteSize();
    TIME_PAINTGL TaskTimer tt("Drawing %u collections (total cache size: %g MB)", N, sumsize/1024.f/1024.f);

    Signal::FinalSource* fs = dynamic_cast<Signal::FinalSource*>(
            model->project()->worker.source()->root());

    TIME_PAINTGL CudaException_CHECK_ERROR();

    model->renderer->init();

    // drawCollections is called for 3 different viewports each frame, don't
    // botter messing around with keeping 3 different frame buffer objects
    // for the different sizes. Recreate the fbo each time instead.
    glEnable( GL_CULL_FACE );

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // Draw the first channel without a frame buffer
    model->renderer->camera = GLvector(model->_qx, model->_qy, model->_qz);
    for (unsigned i=0; i < 1; ++i)
    {
        model->renderer->collection = model->collections[i].get();
        model->renderer->fixed_color = channel_colors[i];
        if (0!=fs)
            fs->set_channel( i );
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
        model->renderer->draw( 1 - orthoview ); // 0.6 ms
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
    }

    if (1<N)
    {
        GlFrameBuffer fbo;
        for (unsigned i=1; i < N; ++i)
        {
            GlException_CHECK_ERROR();
            {
                GlFrameBuffer::ScopeBinding fboBinding = fbo.getScopeBinding();
                glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
                glViewport(0,0,fbo.getGlTexture().getWidth(),fbo.getGlTexture().getHeight());

                model->renderer->collection = model->collections[i].get();
                model->renderer->fixed_color = channel_colors[i];
                if (0!=fs)
                    fs->set_channel( i );
                glDisable(GL_BLEND);
                glEnable(GL_LIGHTING);
                model->renderer->draw( 1 - orthoview ); // 0.6 ms
                glDisable(GL_LIGHTING);
                glEnable(GL_BLEND);
            }
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

            glPushMatrixContext mpc( GL_PROJECTION );
            glLoadIdentity();
            glOrtho(0,1,0,1,-10,10);
            glPushMatrixContext mc( GL_MODELVIEW );
            glLoadIdentity();

            glBlendFunc( GL_DST_COLOR, GL_ZERO );

            glDisable(GL_DEPTH_TEST);

            glColor4f(1,1,1,1);
            GlTexture::ScopeBinding texObjBinding = fbo.getGlTexture().getScopeBinding();
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

    TIME_PAINTGL CudaException_CHECK_ERROR();
    TIME_PAINTGL GlException_CHECK_ERROR();

    glDisable( GL_CULL_FACE );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
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
    glDisable( GL_CULL_FACE ); // enabled only while drawing collections
    glFrontFace( GL_CCW );
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
    glDisable(GL_CULL_FACE);
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
        setPosition( float time, float f )
{
    model->_qx = time;
    model->_qz = f;

    // todo find length by other means

    if (model->_qx<0) model->_qx=0;
    if (model->_qz<0) model->_qz=0;
    if (model->_qz>_last_length) model->_qz=_last_length;
    if (model->_qx>_last_length) model->_qx=_last_length;

    userinput_update();
}


void RenderView::
        makeCurrent()
{
    glwidget->makeCurrent();

    resizeGL(_last_width, _last_height);

    setupCamera();
}


Support::ToolSelector* RenderView::
        toolSelector()
{
//    if (!tool_selector_)
//        tool_selector_.reset( new Support::ToolSelector(glwidget));

    return tool_selector.get();
}


void RenderView::
        userinput_update()
{
    // todo isn't "requested fps" is a renderview property?
    model->project()->worker.requested_fps(60);
    QTimer::singleShot(0, this, SLOT(update())); // this will leave room for others to paint as well, calling 'update' wouldn't
}


void RenderView::
        initializeGL()
{
    //printQGLWidget(*this, "this");
    //TaskTimer("autoBufferSwap=%d", autoBufferSwap()).suppressTiming();
    _inited = true;
}


void RenderView::
        resizeGL( int width, int height )
{
    height = height?height:1;

    glViewport( 0, 0, (GLint)width, (GLint)height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.01f,1000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void RenderView::
        paintGL()
{
    float fps = 0;
    TIME_PAINTGL if (_render_timer)
        fps = 1/_render_timer->elapsedTime();
    TIME_PAINTGL _render_timer.reset();
    TIME_PAINTGL _render_timer.reset(new TaskTimer("Time since last RenderView::paintGL (%g fps)", fps));

    static int tryGc = 0;
    try {
        GlException_CHECK_ERROR();
        CudaException_CHECK_ERROR();


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up camera position
    _last_length = model->project()->worker.source()->length();
    float fs = model->project()->worker.source()->sample_rate();
    {   double limit = std::max(0.f, _last_length - 2*Tfr::Cwt::Singleton().wavelet_time_support_samples(fs)/fs);

        if (model->_qx>=_prevLimit) {
            // -- Following Record Marker --
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            model->_qx = std::max(model->_qx, limit);
        }
        _prevLimit = limit;

        emit prePaint();

        setupCamera();

        glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
        glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
        glGetIntegerv(GL_VIEWPORT, viewport_matrix);
	}

    // TODO move to rendercontroller
    bool wasWorking = !model->project()->worker.todo_list().empty();

    bool onlyUpdateMainRenderView = false;
    { // Render
        if (onlyUpdateMainRenderView)
        foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
        {
            collection->next_frame(); // Discard needed blocks before this row
        }

        drawCollections();

        last_ysize = model->renderer->last_ysize;
        glScalef(1, last_ysize, 1); // global effect on all tools

        emit painting();

        model->renderer->drawAxes( _last_length ); // 4.7 ms

        if (wasWorking)
            Support::DrawWorking::drawWorking( _last_width, _last_height );
    }

    {   // Find things to work on (ie playback and file output)

        //    if (p && p->isUnderfed() && p->invalid_samples_left()) {
        Signal::Intervals missing_in_selection =
                model->project()->tools().playback_model.postsinkCallback->sink()->fetch_invalid_samples();
        if (missing_in_selection)
        {
            model->project()->worker.center = 0;
            model->project()->worker.todo_list( missing_in_selection );

            // Request at least 1 fps. Otherwise there is a risk that CUDA
            // will screw up playback by blocking the OS and causing audio
            // starvation.
            model->project()->worker.requested_fps(1);

            //project->worker.todo_list().print("Displaywidget - PostSink");
        } else {
            model->project()->worker.center = model->_qx;
            model->project()->worker.todo_list(
                    model->collectionCallback->sink()->fetch_invalid_samples());
            //project->worker.todo_list().print("Displaywidget - Collection");
        }
        Signal::Operation* first_source = model->project()->worker.source()->root();
        Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( first_source );
        if(r != 0 && !(r->isStopped()))
        {
            wasWorking = true;
        }
    }

    {   // Work
        bool isWorking = !model->project()->worker.todo_list().empty();

        if (wasWorking || isWorking) {
            if (!_work_timer.get())
                _work_timer.reset( new TaskTimer("Working"));

            // project->worker can be run in one or more separate threads, but if it isn't
            // execute the computations for one chunk
            if (!model->project()->worker.isRunning()) {
                model->project()->worker.workOne();
                QTimer::singleShot(0, this, SLOT(update())); // this will leave room for others to paint as well, calling 'update' wouldn't
            } else {
                //project->worker.todo_list().print("Work to do");
                // Wait a bit while the other thread work
                QTimer::singleShot(200, this, SLOT(update()));

                model->project()->worker.checkForErrors();
            }
        } else {
            static unsigned workcount = 0;
            if (_work_timer) {
                _work_timer->info("Finished %u chunks covering %g s (%g x realtime). Work session #%u",
                                  model->project()->worker.work_chunks,
                                  model->project()->worker.work_time,
                                  model->project()->worker.work_time/_work_timer->elapsedTime(),
                                  workcount);
                model->project()->worker.work_chunks = 0;
                model->project()->worker.work_time = 0;
                workcount++;
                _work_timer.reset();
            }
        }
    }

    if (!onlyUpdateMainRenderView)
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
    {
        // Start looking for which blocks that are requested for the next frame.
        collection->next_frame();
    }

    GlException_CHECK_ERROR();
    CudaException_CHECK_ERROR();

    tryGc = 0;
    } catch (const CudaException &x) {
        TaskTimer tt("RenderView::paintGL CAUGHT CUDAEXCEPTION\n%s", x.what());

        if (2>tryGc) {
            clearCaches();
            tryGc++;
        }
        else throw;
    } catch (const GlException &x) {
        TaskTimer tt("RenderView::paintGL CAUGHT GLEXCEPTION\n%s", x.what());
        if (2>tryGc) {
            clearCaches();
            tryGc++;
        }
        else throw;
    }

    emit postPaint();
}


void RenderView::
        clearCaches()
{
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model->collections )
    {
        Heightmap::Collection* c = collection.get();
        c->reset(); // note, not c.reset()
    }

    Heightmap::Renderer::ColorMode old_color_mode = model->renderer->color_mode;
    model->renderer.reset();
    model->renderer.reset(new Heightmap::Renderer( model->collections[0].get() ));
    model->renderer->color_mode = old_color_mode;
    Tfr::Cwt::Singleton().gc();

    cudaThreadExit();

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
    BOOST_ASSERT( cudaSuccess == e );

    size_t free=0, total=0;

    cudaMemGetInfo(&free, &total);
    TaskInfo("free = %lu, total = %lu", free, total);

    userinput_update();

    cudaGetLastError();
    glGetError();

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

    glScalef(-model->xscale, 1, 5);

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
        channel_colors[i] = channel_colors[i] * (N/2);
    }

    if (1==N)
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
