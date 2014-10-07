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
    Q_PROPERTY(QString displayedTransform READ displayedTransform WRITE setDisplayedTransform NOTIFY displayedTransformChanged)

public:
    Squircle();

    qreal timepos() const;
    void setTimepos(qreal t);

    Chain* chain() const { return chain_item_; }
    void setChain(Chain* c) { chain_item_=c; }

    QString displayedTransform() const { return displayed_transform_; }
    void setDisplayedTransform(QString c);

    Tools::RenderModel* renderModel() { return &render_model; }

signals:
    void timeposChanged();
    void chainChanged();
    void displayedTransformChanged();
    void refresh();

public slots:
    void sync();
    void cleanup();
    void setupUpdateConsumer(QOpenGLContext* context);
    void setupRenderTarget();

private slots:
    void handleWindowChanged(QQuickWindow *win);

protected:
    void componentComplete() override;

private:
    Tools::RenderModel render_model;
    QString displayed_transform_ = "stft";

    Chain* chain_item_ = 0;
    SquircleRenderer *m_renderer = 0;
};

#endif // SQUIRCLE_H
