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
      m_renderer(0),
      touchnavigation(new TouchNavigation(this, &render_model))
{
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
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

    EXCEPTION_ASSERT(chain_item_);

    // requires a window that can be updated
    Tools::Support::RenderViewUpdateAdapter* rvup;
    Tools::Support::RenderOperationDesc::RenderTarget::ptr rvu(
                rvup = new Tools::Support::RenderViewUpdateAdapter);
    rvup->moveToThread (this->thread ());
    connect(rvup, SIGNAL(redraw()), this->window (), SLOT(update())); // render_view, SLOT(redraw()));

    render_model.init(chain_item_->chain (), rvu, chain_item_->target_marker ());
    render_model.render_settings.dpifactor = window()->devicePixelRatio ();

    targetIsCreated();

    // 'this' is parent
    auto hpp = new Tools::Support::HeightmapProcessingPublisher(
                render_model.target_marker (),
                render_model.tfr_mapping (),
                render_model.camera, this);
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
        connect(this, SIGNAL(refresh()), win, SLOT(update()));

        win->setColor (QColor(254,255,255,255));
        win->setClearBeforeRendering(false);
    }
}


void Squircle::
        setChain(Chain* c)
{
    Log("squircle.cpp: setchain %p") % (void*)c;
    chain_item_ = c;
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
        connect(m_renderer, SIGNAL(repositionSignal()), this, SIGNAL(timeposChanged()));
    }

//    m_renderer->setT(m_t);

    QPointF topleft = this->mapToScene (QPointF());
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
    Log("cleanup %d ") % QGLContext::currentContext ();

    if (m_renderer) {
        delete m_renderer;
        m_renderer = 0;
    }
}


void Squircle::componentComplete()
{
    QQuickItem::componentComplete();

    Log("squircle: componentComplete");

    connect(this, SIGNAL(mouseMove(qreal,qreal,bool)),
            touchnavigation, SLOT(mouseMove(qreal,qreal,bool)));
    connect(this, SIGNAL(touch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)),
            touchnavigation, SLOT(touch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)));
    connect(touchnavigation, SIGNAL(refresh()), this, SIGNAL(refresh()));
    connect(touchnavigation, SIGNAL(refresh()), this, SIGNAL(timeposChanged()));
}


void Squircle::targetIsCreated ()
{
    if (QThread::currentThread () != this->thread ())
    {
        // Dispatch
        QMetaObject::invokeMethod (this, "targetIsCreated");
        return;
    }

    RenderViewAxes(render_model).logFreqScale ();
    RenderViewAxes(render_model).logYScale ();
    RenderViewAxes(render_model).cameraOnFront ();
    RenderViewAxes(render_model).logZScale ();
    render_model.render_settings.shadow_shader = true;
}


qreal Squircle::timepos() const
{
    return render_model.camera->q[0];
}


void Squircle::setTimepos (qreal t)
{
    Log("squircle.cpp: t=%g") % t;

    auto c = render_model.camera.write ();
    if (t == c->q[0])
        return;
    c->q[0] = t;
//    emit timeposChanged();
    if (window())
        window()->update();
}
