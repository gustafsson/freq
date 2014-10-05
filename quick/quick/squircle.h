#ifndef SQUIRCLE_H
#define SQUIRCLE_H

#include <QQuickItem>
#include "squirclerenderer.h"
#include "signal/processing/chain.h"
#include "tools/rendermodel.h"
#include "tools/renderview.h"
#include "chain.h"

class Squircle : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(qreal timepos READ timepos WRITE setTimepos NOTIFY timeposChanged)
    Q_PROPERTY(Chain* chain READ chain WRITE setChain NOTIFY chainChanged)
    //Q_PROPERTY(QString Chain* chain READ chain WRITE setChain NOTIFY chainChanged)

public:
    Squircle();

    qreal timepos() const;
    void setTimepos(qreal t);

    Chain* chain() const { return chain_item_; }
    void setChain(Chain* c);// { chain_item_=c; }

signals:
    void timeposChanged();
    void chainChanged();
    void touch(qreal x1, qreal y1, bool p1, qreal x2, qreal y2, bool p2, qreal x3, qreal y3, bool p3);
    void mouseMove(qreal x1, qreal y1, bool p1);
    void refresh();

public slots:
    void sync();
    void cleanup();
    void targetIsCreated();
    void setupUpdateConsumer(QOpenGLContext* context);
    void setupRenderTarget();

private slots:
    void handleWindowChanged(QQuickWindow *win);

protected:
    void componentComplete() override;

private:
    Tools::RenderModel render_model;

    Chain* chain_item_ = 0;
    SquircleRenderer *m_renderer = 0;
    class TouchNavigation* touchnavigation = 0;
};

#endif // SQUIRCLE_H
