#ifndef TOUCHNAVIGATION_H
#define TOUCHNAVIGATION_H

#include <QQuickItem>
#include "tools/rendermodel.h"
#include "squircle.h"
#include "selection.h"

/**
 * @brief The TouchNavigation class should handle touch events on the heightmap
 * for rotating, panning and rescaling
 */
class TouchNavigation : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Squircle* squircle READ squircle WRITE setSquircle NOTIFY squircleChanged)
    Q_PROPERTY(Selection* selection READ selection WRITE setSelection NOTIFY selectionChanged)
    Q_PROPERTY(bool isHold READ isHold NOTIFY isHoldChanged)
public:
    explicit TouchNavigation(QQuickItem* parent=0);

    Squircle* squircle() const { return squircle_; }
    void setSquircle(Squircle*s);

    Selection* selection() const { return selection_; }
    void setSelection(Selection*s);

    bool isHold() const { return is_hold; }

signals:
    void refresh();
    void squircleChanged();
    void selectionChanged();
    void isHoldChanged();

public slots:
    void onTouch(qreal x1, qreal y1, bool p1, qreal x2, qreal y2, bool p2, qreal x3, qreal y3, bool p3);
    void onMouseMove(qreal x1, qreal y1, bool p1);

protected slots:
    void startSelection();

protected:
    void componentComplete() override;

private:
    Tools::RenderModel* render_model();

    Squircle* squircle_ = 0;
    Selection* selection_ = 0;
    bool prevPress1, prevPress2;
    Heightmap::Position hstart1, hstart2;
    QPointF prev1, prev2;
    QPointF start1, start2;
    QTimer timer;
    bool is_hold = false; // use mode instead?

    enum Mode {
        Undefined,
        Rotate,
        Scale
    } mode;
};

#endif // TOUCHNAVIGATION_H
