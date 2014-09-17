#include "squircle.h"
#include "qtmicrophone.h"
#include "signal/recorderoperation.h"
#include "heightmap/update/updateconsumer.h"
#include "tools/support/heightmapprocessingpublisher.h"
#include "tools/support/renderviewupdateadapter.h"
#include "renderviewtransform.h"
#include "renderviewaxes.h"
#include "log.h"
#include "touchnavigation.h"

#include <QQuickWindow>
#include <QTimer>
#include <QOpenGLContext>
#include <QtOpenGL>

class GotDataCallback: public Signal::Recorder::IGotDataCallback
{
public:
    void setInvalidator(Signal::Processing::IInvalidator::ptr i) { i_ = i; }
    void setRecordModel(Squircle* model) { model_ = model; }

    virtual void markNewlyRecordedData(Signal::Interval what)
    {
        if (i_)
            i_->deprecateCache(what);

        if (QQuickWindow* window = model_ ? model_->window () : 0)
            window->update();
    }

private:
    Signal::Processing::IInvalidator::ptr i_;
    Squircle* model_ = 0;
};


Squircle::Squircle() :
      m_t(0),
      m_renderer(0),
      touchnavigation(new TouchNavigation(this, &render_model))
{
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));

    chain = Signal::Processing::Chain::createDefaultChain ();
}


void Squircle::
        setupUpdateConsumer(QOpenGLContext* context)
{
    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        qRegisterMetaType<QOpenGLContext*>("QOpenGLContext*");
        QMetaObject::invokeMethod (this, "setupUpdateConsumer", Q_ARG(QOpenGLContext*, context));
        return;
    }

    // UpdateConsumer shares opengl context with render_view, could use multiple updateconsumers ...
    int n_update_consumers = 1;
    for (int i=0; i<n_update_consumers; i++)
    {
        auto uc = new Heightmap::Update::UpdateConsumer(context, render_model.block_update_queue, 0);
        uc->moveToThread (this->thread ());
        uc->setParent (this);
        connect(uc, SIGNAL(didUpdate()), this->window (), SLOT(update())); //render_view, SLOT(redraw()));
    }
}


void Squircle::
        setupRenderTarget()
{
    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        QMetaObject::invokeMethod (this, "setupRenderTarget");
        return;
    }

    // requires a window that can be updated
    Tools::Support::RenderViewUpdateAdapter* rvup;
    Tools::Support::RenderOperationDesc::RenderTarget::ptr rvu(
                rvup = new Tools::Support::RenderViewUpdateAdapter);
    rvup->moveToThread (this->thread ());
    connect(rvup, SIGNAL(redraw()), this->window (), SLOT(update())); // render_view, SLOT(redraw()));

    render_model.init(chain, rvu);
    render_model.render_settings.dpifactor = window()->devicePixelRatio ();

    targetIsCreated(render_model.target_marker ());

    // 'this' is parent
    auto hpp = new Tools::Support::HeightmapProcessingPublisher(
                render_model.target_marker (),
                render_model.tfr_mapping (),
                &render_model.camera.q[0], this);
    connect(rvup, SIGNAL(setLastUpdatedInterval(Signal::Interval)), hpp, SLOT(setLastUpdatedInterval(Signal::Interval)));

    connect(window(), SIGNAL(afterRendering()), hpp, SLOT(update()));
//    connect(render_view, SIGNAL(painting()), hpp, SLOT(update()));

    RenderViewTransform(render_model).receiveSetTransform_Stft ();
}


void Squircle::handleWindowChanged(QQuickWindow *win)
{
    if (win) {
        connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);
        connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()), Qt::DirectConnection);

        win->setClearBeforeRendering(false);
    }
}


void Squircle::sync()
{
    // Note: The QQuickWindow::beforeSynchronizing() signal is emitted on the rendering
    // thread while the GUI thread is blocked, so it is safe to simply copy the value
    // without any additional protection.

    if (!m_renderer) {
        setupUpdateConsumer(QOpenGLContext::currentContext());
        setupRenderTarget();

        m_renderer = new SquircleRenderer(&render_model);
        connect(window(), SIGNAL(beforeRendering()), m_renderer, SLOT(paint()), Qt::DirectConnection);
        connect(m_renderer, SIGNAL(redrawSignal()), window(), SLOT(update()), Qt::DirectConnection);

        connect(this, SIGNAL(mouseMove(qreal,qreal,bool)),
                touchnavigation, SLOT(mouseMove(qreal,qreal,bool)));
        connect(this, SIGNAL(touch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)),
                touchnavigation, SLOT(touch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)));
        connect(touchnavigation, SIGNAL(refresh()), window(), SLOT(update()));
    }


    m_renderer->setViewportSize(window()->size() * window()->devicePixelRatio());
//    m_renderer->setT(m_t);
}


void Squircle::cleanup()
{
    Log("cleanup %d ") % QGLContext::currentContext ();

    if (m_renderer) {
        delete m_renderer;
        m_renderer = 0;
    }
}


void Squircle::targetIsCreated (Signal::Processing::TargetMarker::ptr target_marker)
{
    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        qRegisterMetaType<Signal::Processing::TargetMarker::ptr>("Signal::Processing::TargetMarker::ptr");
        QMetaObject::invokeMethod (this, "targetIsCreated", Q_ARG(Signal::Processing::TargetMarker::ptr, target_marker));
        return;
    }

    Signal::Recorder::ptr rec(new QtMicrophone);
    GotDataCallback* cb;
    Signal::Recorder::IGotDataCallback::ptr callback(cb = new GotDataCallback());
    Signal::OperationDesc::ptr desc(new Signal::MicrophoneRecorderDesc(rec, callback));
    Signal::Processing::IInvalidator::ptr i = chain->addOperationAt(desc, render_model.target_marker ());
    cb->setInvalidator (i);
    cb->setRecordModel (this);

    rec->startRecording();

    RenderViewAxes(render_model).logFreqScale ();
    RenderViewAxes(render_model).logYScale ();
    RenderViewAxes(render_model).cameraOnFront ();
    render_model.render_settings.shadow_shader = false;
}


void Squircle::setT(qreal t)
{
    if (t == m_t)
        return;
    m_t = t;
    emit tChanged();
    if (window())
        window()->update();
}
