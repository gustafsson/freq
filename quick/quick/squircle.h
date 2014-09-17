#ifndef SQUIRCLE_H
#define SQUIRCLE_H

#include <QQuickItem>
#include "squirclerenderer.h"
#include "signal/processing/chain.h"
#include "tools/rendermodel.h"
#include "tools/renderview.h"

class Squircle : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(qreal t READ t WRITE setT NOTIFY tChanged)

public:
    Squircle();

    qreal t() const { return m_t; }
    void setT(qreal t);

signals:
    void tChanged();

public slots:
    void sync();
    void cleanup();
    void targetIsCreated(Signal::Processing::TargetMarker::ptr target_marker);
    void setupUpdateConsumer(QOpenGLContext* context);
    void setupRenderTarget();

private slots:
    void handleWindowChanged(QQuickWindow *win);

private:
    Tools::RenderModel render_model;

    qreal m_t = 0;
    SquircleRenderer *m_renderer = 0;
    class TouchNavigation* touchnavigation;
    Signal::Processing::Chain::ptr chain;
};

#endif // SQUIRCLE_H
