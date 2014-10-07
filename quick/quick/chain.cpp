#include "chain.h"
#include "log.h"
#include "qtmicrophone.h"
#include "signal/recorderoperation.h"

#include <QtQuick>

using namespace Signal;

class GotDataCallback: public Signal::Recorder::IGotDataCallback
{
public:
    void setInvalidator(Signal::Processing::IInvalidator::ptr i) { i_ = i; }
//    void setRecordModel(Squircle* model) { model_ = model; }

    virtual void markNewlyRecordedData(Signal::Interval what)
    {
        if (i_)
            i_->deprecateCache(what);

//        if (QQuickWindow* window = model_ ? model_->window () : 0)
//            window->update();
    }

private:
    Signal::Processing::IInvalidator::ptr i_;
//    Squircle* model_ = 0;
};


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
    Log("chain.cpp: Created chain %p") % (void*)this;
    openRecording();

    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}


void Chain::handleWindowChanged(QQuickWindow* win)
{
    if (win)
        connect(win, SIGNAL(beforeRendering()), this, SLOT(clearOpenGlBackground()), Qt::DirectConnection);
}


void Chain::clearOpenGlBackground()
{
    // ok as a long as stateless with respect to opengl resources, otherwise this needs a rendering object that is
    // created on window()->beforeSynchronizing and destroyed on window()->sceneGraphInvalidated (as in
    // Squircle/SquircleRenderer)
    glUseProgram (0);
    QColor c = this->window ()->color ();
    glClearColor (c.redF (), c.greenF (), c.blueF (), c.alphaF ());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


void Chain::openRecording()
{
    rec.reset(new QtMicrophone);
    GotDataCallback* cb = new GotDataCallback();
    Signal::Recorder::IGotDataCallback::ptr callback(cb);
    Signal::OperationDesc::ptr desc(new Signal::MicrophoneRecorderDesc(rec, callback));
//    render_model.tfr_mapping ()->channels(desc->extent().number_of_channels.get());
    Signal::Processing::IInvalidator::ptr i = chain_->addOperationAt(desc, target_marker_);
    cb->setInvalidator (i);

    rec->startRecording();
}
