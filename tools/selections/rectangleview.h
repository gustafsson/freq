#ifndef RECTANGLEVIEW_H
#define RECTANGLEVIEW_H

#include <QObject>
#include "signal/worker.h"
#include "rectanglemodel.h"

namespace Tools { namespace Selections
{

class RectangleView: public QObject
{
    Q_OBJECT
public:
    RectangleView(RectangleModel* model, Signal::Worker* worker);
    ~RectangleView();

    void drawSelectionRectangle();
    void drawSelectionRectangle2();

    bool enabled;
    bool visible;

public slots:
    /// Connected in RectangleController
    virtual void draw();

private:
    friend class RectangleController;
    RectangleModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Tools::Selections

#endif // RECTANGLEVIEW_H
