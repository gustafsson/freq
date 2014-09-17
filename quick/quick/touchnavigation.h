#ifndef TOUCHNAVIGATION_H
#define TOUCHNAVIGATION_H

#include <QObject>
#include "tools/rendermodel.h"

/**
 * @brief The TouchNavigation class should handle touch events on the heightmap
 * for rotating, panning and rescaling
 */
class TouchNavigation : public QObject
{
    Q_OBJECT
public:
    TouchNavigation(QObject* parent, Tools::RenderModel* render_model);

public slots:
    void touch(qreal x1, qreal y1, bool p1, qreal x2, qreal y2, bool p2, qreal x3, qreal y3, bool p3);
    void mouseMove(qreal x1, qreal y1, bool p1);

private:
    Tools::RenderModel* render_model;
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
