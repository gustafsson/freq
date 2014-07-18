#include "squircle.h"
#include <QQuickWindow>

Squircle::Squircle()
    : m_t(0)
    , m_renderer(0)
{
    connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
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
        m_renderer = new SquircleRenderer();
        connect(window(), SIGNAL(beforeRendering()), m_renderer, SLOT(paint()), Qt::DirectConnection);
    }
    m_renderer->setViewportSize(window()->size() * window()->devicePixelRatio());
    m_renderer->setT(m_t);
}


void Squircle::cleanup()
{
    if (m_renderer) {
        delete m_renderer;
        m_renderer = 0;
    }
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
