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

// Qt
#include <QTimer>

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

namespace Tools
{

RenderView::
        RenderView(RenderModel* model)
            :
            _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
            _px(0), _py(0), _pz(-10),
            _rx(91), _ry(180), _rz(0),
            xscale(1),
            orthoview(1),
            model(model),
            _work_timer( new TaskTimer("Benchmarking first work"))
{
    float l = model->project()->worker.source()->length();
    _prevLimit = l;

    // TODO remove or explain purpose
    if (_rx<0) _rx=0;
    if (_rx>90) { _rx=90; orthoview=1; }
    if (0<orthoview && _rx<90) { _rx=90; orthoview=0; }
}


RenderView::
        ~RenderView()
{
    TaskTimer tt(__FUNCTION__);

    emit destroying();

    // Because the cuda context was created with cudaGLSetGLDevice it is bound
    // to OpenGL. If we don't have an OpenGL context anymore the Cuda context
    // is corrupt and can't be destroyed nor used properly.
    BOOST_ASSERT( QGLContext::currentContext() );

    // Destroy the cuda context for this thread
    CudaException_SAFE_CALL( cudaThreadExit() );
}


void RenderView::
        setPosition( float time, float f )
{
    _qx = time;
    _qz = f;

    // todo find length by other means
    float l = model->project()->worker.source()->length();

    if (_qx<0) _qx=0;
    if (_qz<0) _qz=0;
    if (_qz>1) _qz=1;
    if (_qx>l) _qx=l;

    // todo isn't "requested fps" is a renderview property?
    userinput_update();
}


Support::ToolSelector* RenderView::
        toolSelector()
{
    if (!_tool_selector)
        _tool_selector.reset( new Support::ToolSelector(this));

    return _tool_selector.get();
}


void RenderView::
        userinput_update()
{
    model->project()->worker.requested_fps(30);
    update();
}

void RenderView::
        initializeGL()
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
    glEnable(GL_COLOR_MATERIAL); // TODO disable?
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
    TIME_PAINTGL _render_timer.reset();
    TIME_PAINTGL _render_timer.reset(new TaskTimer("Time since last RenderView::paintGL"));

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

        if (_qx>=_prevLimit) {
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            _qx = std::max(_qx, limit);
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

        model->renderer->camera = GLvector(_qx, _qy, _qz);
        model->renderer->draw( 1 - orthoview ); // 0.6 ms

        emit painting();

        model->renderer->drawAxes( length ); // 4.7 ms

        if (wasWorking)
            Support::DrawWorking::drawWorking(width(), height());
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
            model->project()->worker.center = _qx;
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
                _work_timer->info("Finished %u chunks covering %g s. Work session #%u",
                                  model->project()->worker.work_chunks,
                                  model->project()->worker.work_time, workcount);
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
        if (0==tryGc) {
            model->collection->gc();
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
    glTranslatef( _px, _py, _pz );

    glRotatef( _rx, 1, 0, 0 );
    glRotatef( fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview, 0, 1, 0 );
    glRotatef( _rz, 0, 0, 1 );

    glScalef(-xscale, 1, 5);

    glTranslatef( -_qx, -_qy, -_qz );

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
