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
    Q_PROPERTY(qreal timepos READ timepos WRITE setTimepos NOTIFY cameraChanged)
    Q_PROPERTY(qreal scalepos READ scalepos WRITE setScalepos NOTIFY cameraChanged)
    Q_PROPERTY(qreal xscale READ xscale WRITE setXscale NOTIFY cameraChanged)
    Q_PROPERTY(qreal xangle READ xangle WRITE setXangle NOTIFY cameraChanged)
    Q_PROPERTY(Chain* chain READ chain WRITE setChain NOTIFY chainChanged)
    Q_PROPERTY(QString displayedTransform READ displayedTransform WRITE setDisplayedTransform NOTIFY displayedTransformChanged)
    Q_PROPERTY(QString displayedTransformDetails READ displayedTransformDetails NOTIFY displayedTransformDetailsChanged)
    Q_PROPERTY(bool isIOS READ isIOS CONSTANT)

public:
    Squircle();
    ~Squircle();

    qreal timepos() const;
    void setTimepos(qreal v);

    qreal scalepos() const;
    void setScalepos(qreal v);

    qreal xscale() const;
    void setXscale(qreal v);

    qreal xangle() const;
    void setXangle(qreal v);

    Chain* chain() const { return chain_item_; }
    void setChain(Chain* c);

    QString displayedTransform() const { return displayed_transform_; }
    void setDisplayedTransform(QString c);
    QString displayedTransformDetails() const;

    bool isIOS() const;

    Tools::RenderModel* renderModel() { return &render_model; }

signals:
    void cameraChanged();
    void chainChanged();
    void displayedTransformChanged();
    void displayedTransformDetailsChanged();
    void refresh();

    // use Qt::DirectConnection to ensure that objects created are placed on the same thread
    void rendererChanged(SquircleRenderer* renderer);


public slots:
    void sync();
    void cleanup();
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
