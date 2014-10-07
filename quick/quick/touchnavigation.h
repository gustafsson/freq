#ifndef TOUCHNAVIGATION_H
#define TOUCHNAVIGATION_H

#include <QQuickItem>
#include "tools/rendermodel.h"

/**
 * @brief The TouchNavigation class should handle touch events on the heightmap
 * for rotating, panning and rescaling
 */
class TouchNavigation : public QQuickItem
{
    Q_OBJECT
public:
    explicit TouchNavigation(QQuickItem* parent=0);

signals:
    void refresh();

public slots:
    void onTouch(qreal x1, qreal y1, bool p1, qreal x2, qreal y2, bool p2, qreal x3, qreal y3, bool p3);
    void onMouseMove(qreal x1, qreal y1, bool p1);

protected:
    void componentComplete() override;
    void itemChange(ItemChange change, const ItemChangeData & value) override;

private:
    Tools::RenderModel* render_model = 0;
    bool prevPress1, prevPress2;
    Heightmap::Position hstart1, hstart2;
    QPointF prev1, prev2;
    QPointF start1, start2;

    enum Mode {
        Undefined,
        Rotate,
        Scale
    } mode;
};

#endif // TOUCHNAVIGATION_H
