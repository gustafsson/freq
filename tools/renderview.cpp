#include "renderview.h"

// TODO cleanup

// Sonic AWE
#include "sawe/project.h"
#include "tfr/cwt.h"
#include "toolfactory.h"
#include "support/drawworking.h"
#include "adapters/microphonerecorder.h"

// gpumisc
#include <CudaException.h>
#include <GlException.h>
#include <glPushContext.h>
#include <demangle.h>

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

	// Validate rotation and set orthoview accordingly
    if (model->_rx<0) model->_rx=0;
    if (model->_rx>90) { model->_rx=90; orthoview=1; }
    if (0<orthoview && model->_rx<90) { model->_rx=90; orthoview=0; }
}


RenderView::
        ~RenderView()
{
    TaskTimer tt(__FUNCTION__);

    emit destroying();

    // Because the cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
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

        if (!_inited)
            initializeGL();

        _last_width = painter->device()->width();
        _last_height = painter->device()->height();

        setStates();
        resizeGL(_last_width, _last_height);

        paintGL();

        {
            glPushMatrixContext pmc(GL_MODELVIEW);

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
        getHeightmapValue( Heightmap::Position pos )
{
    if (pos.time < 0 || pos.scale < 0 || pos.scale > 1)
        return 0;

    Heightmap::Reference ref = model->renderer->findRefAtCurrentZoomLevel( pos.time, pos.scale );
    Heightmap::pBlock block = model->collection->getBlock( ref );
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();

    float* data = blockData->getCpuMemory();
    Heightmap::Position a,b;
    ref.getArea( a, b );
    unsigned w = ref.samplesPerBlock();
    unsigned h = ref.scalesPerBlock();
    unsigned x0 = (pos.time-a.time)/(b.time-a.time)*(w-1) + .5f;
    unsigned y0 = (pos.scale-a.scale)/(b.scale-a.scale)*(h-1) + .5f;

    float v = data[ x0 + y0*w ];
    blockData->getCudaGlobal(false);

    v *= model->renderer->y_scale;
    v *= 4;
    return v;
}


QPointF RenderView::
        getScreenPos( Heightmap::Position pos, double* dist )
{
    GLdouble objY = getHeightmapValue(pos);
    GLdouble winX, winY, winZ;
    gluProject( pos.time, objY, pos.scale,
                m, proj, vp,
                &winX, &winY, &winZ);

    if (dist)
    {
        float z0 = .1, z1=.2;
        GLvector projectionPlane = Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z0) );
        GLvector projectionNormal = (Heightmap::gluUnProject( GLvector( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z1) ) - projectionPlane).Normalize();

        GLvector p;
        p[0] = pos.time;
        p[1] = 0;//objY;
        p[2] = pos.scale;

        *dist = (p-projectionPlane)%projectionNormal;
        *dist *= last_ysize;
    }

    return QPointF( winX, vp[3]-1-winY );
    //return QPointF( winX, winY );
}


Heightmap::Position RenderView::
        getHeightmapPos( QPointF pos )
{
    TaskTimer tt("RenderView::getPlanePos Newton raphson");

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

        y = getHeightmapValue(p);
        tt.info("(%g, %g) %g", p.time, p.scale, y);
    }
    return p;
}

Heightmap::Position RenderView::
        getPlanePos( QPointF pos, bool* success )
{
    GLdouble m[16], proj[16];
    GLint vp[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, m);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, vp);

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

    *success=true;
    float minAngle = 3;
    if (0) if(success)
    {
        if( s < 0 )
            *success=false;

        float L = sqrt((objX1-objX2)*(objX1-objX2)
                       +(objY1-objY2)*(objY1-objY2)
                       +(objZ1-objZ2)*(objZ1-objZ2));
        if (objY1-objY2 < model->xscale*sin(minAngle *(M_PI/180)) * L )
            *success=false;
    }

    return p;
}


void RenderView::
        setStates()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glShadeModel(GL_SMOOTH);

    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    //glEnable(GL_CULL_FACE);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    {   // Antialiasing
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        glEnable(GL_BLEND);
    }


    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL); // TODO disable?

    //glDisable(GL_COLOR_MATERIAL); // Must disable texturing as well when drawing primitives
    //glEnable(GL_TEXTURE_2D);
    //glEnable(GL_NORMALIZE);

    setLights();

    //float materialSpecular[] = {0.5f, 0.5f, 0.5f, 1.0f};
    //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, materialSpecular);
    //glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0f);
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

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);

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
    float l = model->project()->worker.source()->length();

    if (model->_qx<0) model->_qx=0;
    if (model->_qz<0) model->_qz=0;
    if (model->_qz>1) model->_qz=1;
    if (model->_qx>l) model->_qx=l;

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
    update();
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
    bool followingRecordMarker = false;
    float length = model->project()->worker.source()->length();
    float fs = model->project()->worker.source()->sample_rate();
    {   double limit = std::max(0.f, length - 2*Tfr::Cwt::Singleton().wavelet_time_support_samples(fs)/fs);

        if (model->_qx>=_prevLimit) {
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            model->_qx = std::max(model->_qx, limit);
            followingRecordMarker = true;
        }
        _prevLimit = limit;

        emit prePaint();

        setupCamera();
    }

    // TODO move to rendercontroller
    bool wasWorking = !model->project()->worker.todo_list().empty();

    { // Render
        model->collection->next_frame(); // Discard needed blocks before this row

        model->renderer->camera = GLvector(model->_qx, model->_qy, model->_qz);
        model->renderer->draw( 1 - orthoview ); // 0.6 ms

        last_ysize = model->renderer->last_ysize;
        glGetDoublev(GL_MODELVIEW_MATRIX, m);
        glGetDoublev(GL_PROJECTION_MATRIX, proj);
        glGetIntegerv(GL_VIEWPORT, vp);

        emit painting();

        model->renderer->drawAxes( length ); // 4.7 ms

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

            if (followingRecordMarker)
                model->project()->worker.requested_fps(1);
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

    GlException_CHECK_ERROR();
    CudaException_CHECK_ERROR();

    tryGc = 0;
    } catch (const CudaException &x) {
        TaskTimer tt("RenderView::paintGL CAUGHT CUDAEXCEPTION\n%s", x.what());
        if (2>tryGc) {
            Heightmap::Collection* c = model->collection.get();
            c->reset(); // note, not c.reset()
            model->renderer.reset();
            model->renderer.reset(new Heightmap::Renderer( c ));
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
        TaskTimer tt("RenderView::paintGL CAUGHT GLEXCEPTION\n%s", x.what());
        if (2>tryGc) {
            Heightmap::Collection* c = model->collection.get();
            c->reset(); // note, not c.reset()
            model->renderer.reset();
            model->renderer.reset(new Heightmap::Renderer( c ));
            tryGc++;
            //cudaThreadExit();
            cudaGetLastError();
        }
        else throw;
    }

    emit postPaint();
}


void RenderView::
        setupCamera()
{
    glLoadIdentity();
    glTranslatef( model->_px, model->_py, model->_pz );

    glRotatef( model->_rx, 1, 0, 0 );
    glRotatef( fmod(fmod(model->_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(model->_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( model->_rz, 0, 0, 1 );

    glScalef(-model->xscale, 1, 5);

    glTranslatef( -model->_qx, -model->_qy, -model->_qz );

    orthoview.TimeStep(.08);
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
