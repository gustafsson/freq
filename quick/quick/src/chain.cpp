#include "chain.h"
#include "log.h"
#include "qtmicrophone.h"
#include "signal/recorderoperation.h"
#include "heightmap/update/updateconsumer.h"

#include <QtQuick>

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
    update_queue_->abort_on_empty();
    update_queue_->clear ();
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
        connect(win, SIGNAL(beforeRendering()), this, SLOT(clearOpenGlBackground()), Qt::DirectConnection);
}


void setStates(tvector<4,float> a)
{
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
}


void Chain::clearOpenGlBackground()
{
    if (!update_consumer_)
        setupUpdateConsumer(QOpenGLContext::currentContext());

    // ok as a long as stateless with respect to opengl resources, otherwise this needs a rendering object that is
    // created on window()->beforeSynchronizing and destroyed on window()->sceneGraphInvalidated (as in
    // Squircle/SquircleRenderer)
    glUseProgram (0);
    QColor c = this->window ()->color ();
    setStates(tvector<4,float>(c.redF (),c.greenF (),c.blueF (),c.alphaF ()));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


void Chain::openRecording()
{
    Signal::Recorder::ptr rec(new QtMicrophone);
    Signal::OperationDesc::ptr desc(new Signal::MicrophoneRecorderDesc(rec));
    Signal::Processing::IInvalidator::ptr i = chain_->addOperationAt(desc, target_marker_);

    rec->setInvalidator( i );
    rec->startRecording();

    setTitle (rec->name().c_str());
}


void Chain::setupUpdateConsumer(QOpenGLContext* context)
{
    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        qRegisterMetaType<QOpenGLContext*>("QOpenGLContext*");
        QMetaObject::invokeMethod (this, "setupUpdateConsumer", Q_ARG(QOpenGLContext*, context));
        return;
    }

    // UpdateConsumer shares OpenGL context and is owned by this
    update_consumer_ = new Heightmap::Update::UpdateConsumer(context, update_queue_, this);
    connect(update_consumer_.data (), SIGNAL(didUpdate()), this->window (), SLOT(update()));
}
