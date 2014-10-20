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
    Q_PROPERTY(qreal scalepos READ scalepos WRITE setScalepos NOTIFY scaleposChanged)
    Q_PROPERTY(qreal timezoom READ timezoom WRITE setTimezoom NOTIFY timezoomChanged)
    Q_PROPERTY(qreal scalezoom READ scalezoom WRITE setScalezoom NOTIFY scalezoomChanged)
    Q_PROPERTY(qreal xangle READ xangle WRITE setXangle NOTIFY xangleChanged)
    Q_PROPERTY(qreal yangle READ yangle WRITE setYangle NOTIFY yangleChanged)
    Q_PROPERTY(Chain* chain READ chain WRITE setChain NOTIFY chainChanged)
    Q_PROPERTY(QString displayedTransform READ displayedTransform WRITE setDisplayedTransform NOTIFY displayedTransformChanged)
    Q_PROPERTY(QString displayedTransformDetails READ displayedTransformDetails NOTIFY displayedTransformDetailsChanged)
    Q_PROPERTY(QString displayedHeight READ displayedHeight WRITE setDisplayedHeight NOTIFY displayedHeightChanged)
    Q_PROPERTY(float equalizeColors READ equalizeColors WRITE setEqualizeColors NOTIFY equalizeColorsChanged)
    Q_PROPERTY(bool isIOS READ isIOS CONSTANT)

public:
    Squircle();
    ~Squircle();

    qreal timepos() const;
    void setTimepos(qreal v);

    qreal scalepos() const;
    void setScalepos(qreal v);

    qreal timezoom() const;
    void setTimezoom(qreal v);

    qreal scalezoom() const;
    void setScalezoom(qreal v);

    qreal xangle() const;
    void setXangle(qreal v);

    qreal yangle() const;
    void setYangle(qreal v);

    Chain* chain() const { return chain_item_; }
    void setChain(Chain* c);

    QString displayedTransform() const { return displayed_transform_; }
    void setDisplayedTransform(QString c);
    QString displayedTransformDetails() const;

    QString displayedHeight() const { return displayed_height_; }
    void setDisplayedHeight(QString c);

    float equalizeColors();
    void setEqualizeColors(float);

    bool isIOS() const;

    Tools::RenderModel* renderModel() { return &render_model; }

signals:
    void timeposChanged();
    void scaleposChanged();
    void timezoomChanged();
    void scalezoomChanged();
    void xangleChanged();
    void yangleChanged();
    void chainChanged();
    void displayedTransformChanged();
    void displayedTransformDetailsChanged();
    void displayedHeightChanged();
    void equalizeColorsChanged();

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
    QString displayed_height_ = "log";

    Chain* chain_item_ = 0;
    SquircleRenderer *m_renderer = 0;
};

#endif // SQUIRCLE_H
