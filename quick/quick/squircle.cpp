#include "squircle.h"

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

Squircle::Squircle() :
      m_renderer(0)
{
    connect (this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
}


Squircle::~Squircle()
{
    cleanup();
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

    EXCEPTION_ASSERT(chain_item_);

    // requires a window that can be updated
    Tools::Support::RenderViewUpdateAdapter* rvup;
    Tools::Support::RenderOperationDesc::RenderTarget::ptr rvu(
                rvup = new Tools::Support::RenderViewUpdateAdapter);
    rvup->moveToThread (this->thread ());
    connect(rvup, SIGNAL(redraw()), this->window (), SLOT(update())); // render_view, SLOT(redraw()));

    render_model.init(chain_item_->chain (), chain_item_->update_queue (), rvu, chain_item_->target_marker ());
    render_model.render_settings.dpifactor = window()->devicePixelRatio ();

    double prio = displayedTransform() == "waveform" ? 2 : 0;

    // 'this' is parent
    auto hpp = new Tools::Support::HeightmapProcessingPublisher(
                render_model.target_marker (),
                render_model.tfr_mapping (),
                render_model.camera, prio, this);
    connect(rvup, SIGNAL(setLastUpdatedInterval(Signal::Interval)), hpp, SLOT(setLastUpdatedInterval(Signal::Interval)));

    connect(window(), SIGNAL(afterRendering()), hpp, SLOT(update()));
//    connect(render_view, SIGNAL(painting()), hpp, SLOT(update()));

    RenderViewAxes(render_model).logYScale ();
    RenderViewAxes(render_model).cameraOnFront ();
    RenderViewAxes(render_model).logZScale ();
    render_model.render_settings.shadow_shader = true;
    render_model.camera->xscale = 10;

    setDisplayedTransform(displayedTransform());
}


void Squircle::handleWindowChanged(QQuickWindow *win)
{
    if (win) {
        disconnect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()));
        disconnect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()));
        disconnect(this, SIGNAL(cameraChanged()), this, SIGNAL(refresh()));
        disconnect(this, SIGNAL(refresh()), win, SLOT(update()));
        connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);
        connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()), Qt::DirectConnection);
        connect(this, SIGNAL(cameraChanged()), SIGNAL(refresh()));
        connect(this, SIGNAL(refresh()), win, SLOT(update()));

        auto v = render_model.render_settings.clear_color;
        auto f = [](float v) { return (unsigned char)(v<0.f?0:v>1.f?255:v*255); };
        QColor c(f(v[0]), f(v[1]), f(v[2]), f(v[3]));
        win->setColor (c);
        win->setClearBeforeRendering (false);

        QMetaObject::invokeMethod (win, "update", Qt::QueuedConnection);
    }
}


void Squircle::
        setDisplayedTransform(QString c)
{
    if (m_renderer) {
        if (c == "stft")
            RenderViewTransform(render_model).receiveSetTransform_Stft ();
        else if (c == "waveform")
        {
            RenderViewTransform(render_model).receiveSetTransform_Waveform ();
            RenderViewAxes(render_model).waveformScale ();
            render_model.resetCameraSettings ();
        }
        else
            Log("squircle: unrecognized transform string: \"%s\"") % c.toStdString ();
    }

    emit displayedTransformDetailsChanged();

    if (displayed_transform_ == c)
        return;

    displayed_transform_ = c;
    emit displayedTransformChanged();
}


QString Squircle::
        displayedTransformDetails() const
{
    if (auto t = render_model.transform_desc ())
        return QString::fromStdString (t->toString ());
    return "[no transform]";
}


bool Squircle::
        isIOS() const
{
#ifdef Q_OS_IOS
    return true;
#else
    return false;
#endif
}


void Squircle::sync()
{
    if (!isVisible () && m_renderer) {
        delete m_renderer;
        m_renderer = 0;
    }

    if (isVisible () && !m_renderer) {
        m_renderer = new SquircleRenderer(&render_model);
        m_renderer->setObjectName (objectName () + " renderer");
        connect(window(), SIGNAL(beforeRendering()), m_renderer, SLOT(paint()), Qt::DirectConnection);
        connect(m_renderer, SIGNAL(redrawSignal()), window(), SLOT(update()));
        connect(m_renderer, SIGNAL(repositionSignal()), this, SIGNAL(cameraChanged()));

        emit rendererChanged(m_renderer);

        setupRenderTarget();
    }

    if (!m_renderer)
        return;

    // Note: The QQuickWindow::beforeSynchronizing() signal is emitted on the rendering
    // thread while the GUI thread is blocked, so it is safe to simply copy the value
    // without any additional protection.
//    m_renderer->setT(m_t);

    QPointF topleft = this->mapToScene (boundingRect().topLeft ());
    QPointF bottomright = this->mapToScene (boundingRect().bottomRight ());

    if (0) Log("squircle: sync x1=%g, y1=%g, x2=%g, y2=%g, w=%g, h=%g")
            % topleft.x () % topleft.y ()
            % bottomright.x () % bottomright.y ()
            % width () % height();

    m_renderer->setViewport(QRectF(topleft, bottomright),
                            window ()->height (), window()->devicePixelRatio());
}


void Squircle::cleanup()
{
    if (m_renderer) {
        Log("squircle: cleanup");

        delete m_renderer;
        m_renderer = 0;
    }
}


void Squircle::componentComplete()
{
    QQuickItem::componentComplete();

    if (window ())
        handleWindowChanged(window ());
}


qreal Squircle::timepos() const
{
    return render_model.camera.read ()->q[0];
}


void Squircle::setTimepos (qreal v)
{
    auto c = render_model.camera.write ();
    if (v == qreal(c->q[0]))
        return;
    c->q[0] = v;
    c.unlock ();

    emit cameraChanged ();
}


qreal Squircle::scalepos() const
{
    return render_model.camera.read ()->q[2];
}


void Squircle::setScalepos(qreal v)
{
    auto c = render_model.camera.write ();
    if (v == qreal(c->q[2]))
        return;
    c->q[2] = v;
    c.unlock ();

    emit cameraChanged ();
}


qreal Squircle::xscale() const
{
    return render_model.camera.read ()->xscale;
}


void Squircle::setXscale(qreal v)
{
    auto c = render_model.camera.write ();
    if (v == qreal(c->xscale))
        return;
    c->xscale = v;
    c.unlock ();

    emit cameraChanged ();
}


qreal Squircle::xangle() const
{
    return render_model.camera.read ()->r[0];
}


void Squircle::setXangle(qreal v)
{
    auto c = render_model.camera.write ();
    if (v == qreal(c->r[0]))
        return;
    c->r[0] = v;
    c.unlock ();

    emit cameraChanged ();
}


qreal Squircle::yangle() const
{
    return render_model.camera.read ()->r[1];
}


void Squircle::setYangle(qreal v)
{
    auto c = render_model.camera.write ();
    if (v == qreal(c->r[1]))
        return;
    c->r[1] = v;
    c.unlock ();

    emit cameraChanged ();
}


void Squircle::
        setChain(Chain* c)
{
    if (c==chain_item_)
        return;
    chain_item_=c;
    emit chainChanged();
}
