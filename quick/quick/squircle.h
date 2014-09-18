#ifndef SQUIRCLE_H
#define SQUIRCLE_H

#include <QQuickItem>
#include "squirclerenderer.h"
#include "signal/processing/chain.h"
#include "tools/rendermodel.h"
#include "tools/renderview.h"
#include "signal/recorder.h"

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
    void targetIsCreated();
    void setupUpdateConsumer(QOpenGLContext* context);
    void setupRenderTarget();
    void urlRequest(QUrl url);

private slots:
    void handleWindowChanged(QQuickWindow *win);

private:
    void openUrl(QUrl url);
    void openRecording();
    void purgeTarget();

    Tools::RenderModel render_model;

    qreal m_t = 0;
    SquircleRenderer *m_renderer = 0;
    class TouchNavigation* touchnavigation;
    Signal::Processing::Chain::ptr chain;
    QUrl url;
    Signal::Recorder::ptr rec;
};

#endif // SQUIRCLE_H
