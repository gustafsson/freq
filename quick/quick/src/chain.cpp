#include "chain.h"
#include "log.h"
#include "qtmicrophone.h"
#include "signal/recorderoperation.h"
#include "signal/qteventworker/bedroomsignaladapter.h"
#include "heightmap/update/updateconsumer.h"
#include "heightmap/uncaughtexception.h"
#include "GlException.h"
#include "demangle.h"
#include "glgroupmarker.h"
#include "tasktimer.h"
#include "glstate.h"

#include <QtQuick>

//#define CHAIN_USEUPDATECONSUMERTHREAD

using namespace Signal;

class NoopOperationImpl: public Operation
{
public:
    pBuffer process(pBuffer b) override {
        return b;
    }
};


class NoopOperation: public OperationDesc
{
public:
    Interval requiredInterval( const Interval& I, Interval* expectedOutput ) const override {
        if (expectedOutput)
            *expectedOutput = I;
        return I;
    }

    Interval affectedInterval( const Interval& I ) const override {
        return I;
    }

    OperationDesc::ptr copy() const override {
        return OperationDesc::ptr(new NoopOperation);
    }

    Operation::ptr createOperation(ComputingEngine*) const override {
        return Operation::ptr(new NoopOperationImpl);
    }
};


Chain::Chain(QQuickItem *parent) :
    QQuickItem(parent)
{
    chain_ = Processing::Chain::createDefaultChain ();
    target_marker_ = chain_->addTarget(OperationDesc::ptr(new NoopOperation));
    update_queue_.reset (new Heightmap::Update::UpdateQueue);

    openRecording();

    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}


Chain::
        ~Chain()
{
    if (chain_)
        sceneGraphInvalidated ();
}


void Chain::
        setTitle(QString v)
{
    if (title_==v)
        return;
    title_ = v;
    emit titleChanged();
}


void Chain::handleWindowChanged(QQuickWindow* win)
{
    if (win)
    {
        connect(win, SIGNAL(beforeRendering()), this, SLOT(clearOpenGlBackground()), Qt::DirectConnection);
        connect(win, SIGNAL(afterRendering()), this, SLOT(afterRendering()), Qt::DirectConnection);
        connect(win, SIGNAL(sceneGraphInitialized()), this, SLOT(sceneGraphInitialized()), Qt::DirectConnection);
        connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(sceneGraphInvalidated()), Qt::DirectConnection);
    }
}


void setStates()
{
    GlState::lost_sync();

#ifdef GL_ES_VERSION_2_0
    GlException_SAFE_CALL( glClearDepthf(1.0f) );
#else
    GlException_SAFE_CALL( glClearDepth(1.0) );
#endif

#ifdef LEGACY_OPENGL
    GlException_SAFE_CALL( glEnable(GL_TEXTURE_2D) );
#endif

    GlException_SAFE_CALL( glDepthMask(true) );

    GlException_SAFE_CALL( glEnable(GL_DEPTH_TEST) );
    GlException_SAFE_CALL( glDepthFunc(GL_LEQUAL) );
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
#ifdef LEGACY_OPENGL

    GlException_SAFE_CALL( glShadeModel(GL_SMOOTH) );
    GlException_SAFE_CALL( glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST) );

    // Antialiasing
    // This is not a recommended method for anti-aliasing. Use Multisampling instead.
    // https://www.opengl.org/wiki/Common_Mistakes#glEnable.28GL_POLYGON_SMOOTH.29
    //GlException_SAFE_CALL( glEnable(GL_LINE_SMOOTH) );
    //GlException_SAFE_CALL( glHint(GL_LINE_SMOOTH_HINT, GL_NICEST) );
    //GlException_SAFE_CALL( glEnable(GL_POLYGON_SMOOTH) );
    //GlException_SAFE_CALL( glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST) );
    //GlException_SAFE_CALL( glDisable(GL_POLYGON_SMOOTH) );
#endif

    GlException_SAFE_CALL( glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) );
    GlException_SAFE_CALL( glEnable(GL_BLEND) );
}


void Chain::clearOpenGlBackground()
{
    static bool failed = false;
    if (failed)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

    try {

    GlGroupMarker gpm("clearOpenGlBackground");

    render_timer.restart ();

#ifndef LEGACY_OPENGL
    if (!vertexArray_)
        GlException_SAFE_CALL( glGenVertexArrays(1, &vertexArray_) );
    GlException_SAFE_CALL( glBindVertexArray(vertexArray_) );
#endif

#ifndef CHAIN_USEUPDATECONSUMERTHREAD
    update_consumer_p->workIfAny();
#endif

    // ok as a long as stateless with respect to opengl resources, otherwise this needs a rendering object that is
    // created on window()->beforeSynchronizing and destroyed on window()->sceneGraphInvalidated (as in
    // Squircle/SquircleRenderer)
    setStates();
    QColor c = this->window ()->color ();
    GlException_SAFE_CALL( glClearColor(c.redF (), c.greenF (), c.blueF (), c.alphaF ()) );
    GlException_SAFE_CALL( glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) );

    } catch (const std::exception& x) {
        fflush(stdout);
        fprintf(stderr, "%s",
                (boost::format("%s\n"
                                  "%s\n"
                                  " FAILED in %s\n\n")
                    % vartype(x) % boost::diagnostic_information(x) % __FUNCTION__ ).str().c_str());
        fflush(stderr);
        failed = true;
    }
}


void Chain::afterRendering()
{
    render_time += render_timer.elapsed ();
    if (ltf.tick(false))
    {
        Log("renderchain: %g frames/s, activity %.0f%%") % ltf.hz () % (100*render_time/start_timer.elapsedAndRestart ());
        render_time = 0;
    }

    if (auto r = rec_.lock ())
    {
        // keep animating during recording with a full framerate (60 fps) even if tasks are finished at a lower rate
        if (!r->isStopped()) {
            auto c = chain()->targets()->getTargets();
            for (const auto& t : c) {
                Signal::IntervalType i = r->number_of_samples();
                if (t->needed() & Signal::Interval(i,i+1))
                    QMetaObject::invokeMethod (window(), "update");
            }
        }
    }
}


void Chain::sceneGraphInitialized()
{
    TaskInfo ti("Chain::sceneGraphInitialized()");
    EXCEPTION_ASSERT(QOpenGLContext::currentContext ());

#ifdef CHAIN_USEUPDATECONSUMERTHREAD
    if (!update_consumer_thread_)
        setupUpdateConsumerThread(QOpenGLContext::currentContext());
#else
    if (!update_consumer_p)
    {
        update_consumer_p.reset (new Heightmap::Update::UpdateConsumer(update_queue_));
        setupBedroomUpdateThread();
    }
#endif
}


void Chain::sceneGraphInvalidated()
{
    TaskInfo ti("Chain::sceneGraphInvalidated() %p", QOpenGLContext::currentContext());
    update_queue_->close ();
    chain_->close ();

    update_consumer_p.reset ();

    if (update_consumer_thread_)
    {
        delete update_consumer_thread_;
        update_consumer_thread_ = 0;
    }

    chain_.reset ();
}


void Chain::openRecording()
{
    Signal::Recorder::ptr rec(new QtMicrophone);
    Signal::OperationDesc::ptr desc(new Signal::MicrophoneRecorderDesc(rec));
    Signal::Processing::IInvalidator::ptr i = chain_->addOperationAt(desc, target_marker_);

    rec->setInvalidator( i );
    rec->startRecording();

    setTitle (rec->name().c_str());

    rec_ = rec;
}


void Chain::setupUpdateConsumerThread(QOpenGLContext* context)
{
    EXCEPTION_ASSERT(!update_consumer_p);

    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        qRegisterMetaType<QOpenGLContext*>("QOpenGLContext*");
        QMetaObject::invokeMethod (this, "setupUpdateConsumerThread", Q_ARG(QOpenGLContext*, context));
        return;
    }

    // UpdateConsumerThread shares OpenGL context and is owned by this
    update_consumer_thread_ = new Heightmap::Update::UpdateConsumerThread(context, update_queue_, this);
    connect(update_consumer_thread_.data (), SIGNAL(didUpdate()), this->window (), SLOT(update()));
}


void Chain::setupBedroomUpdateThread()
{
    EXCEPTION_ASSERT(!update_consumer_thread_);

    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        QMetaObject::invokeMethod (this, "setupBedroomUpdateThread");
        return;
    }

    // BedroomSignalAdapter is owned by this
    auto b = new Signal::QtEventWorker::BedroomSignalAdapter(chain_->bedroom(), this);
    connect(b, SIGNAL(wakeup()), this->window (), SLOT(update()));
}
